import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class FP8Config:
    total_bits: int = 8
    sign_bits: int = 1

class GradientQuartileTracker:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.q1: Optional[torch.Tensor] = None
        self.q2: Optional[torch.Tensor] = None
        self.q3: Optional[torch.Tensor] = None
        self.initialized = False

    def update(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_abs = gradients.abs().flatten()

        if grad_abs.sum() == 0:
            if self.initialized:
                return self.q1, self.q2, self.q3
            else:
                device = gradients.device
                self.q1 = torch.tensor(0.0, device=device)
                self.q2 = torch.tensor(0.0, device=device)
                self.q3 = torch.tensor(0.0, device=device)
                self.q19 = torch.tensor(0.0, device=device)
                return self.q1, self.q2, self.q3

        current_q1 = torch.quantile(grad_abs, 0.25)
        current_q2 = torch.quantile(grad_abs, 0.50)
        current_q3 = torch.quantile(grad_abs, 0.75)

        if not self.initialized:
            self.q1 = current_q1
            self.q2 = current_q2
            self.q3 = current_q3
            self.initialized = True
        else:
            self.q1 = self.momentum * self.q1 + (1 - self.momentum) * current_q1
            self.q2 = self.momentum * self.q2 + (1 - self.momentum) * current_q2
            self.q3 = self.momentum * self.q3 + (1 - self.momentum) * current_q3

        return self.q1, self.q2, self.q3

    def get_quartiles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.initialized:
            raise RuntimeError("Quartiles not initialized. Call update() first.")
        return self.q1, self.q2, self.q3

class DynamicFP8Quantizer(nn.Module):
    def __init__(self, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.quartile_tracker = GradientQuartileTracker(momentum=0.9)

        if n_bits == 8:
            self.exp_bits_per_region = [2, 3, 5, 6]
        elif n_bits == 4:
            self.exp_bits_per_region = [0, 1, 2, 2]
        else:
            self.exp_bits_per_region = None

    def _quantize_to_fp_format(
        self,
        x: torch.Tensor,
        exp_bits: int,
        mantissa_bits: int
    ) -> torch.Tensor:
        if exp_bits == 0:
            n_levels = 1 << mantissa_bits
            x_max = x.abs().max()
            scale = x_max / (n_levels - 1) if x_max > 0 else 1.0
            return torch.round(x / scale) * scale

        bias = (1 << (exp_bits - 1)) - 1
        max_exp = (1 << exp_bits) - 1 - bias

        max_mantissa = 1.0 + (1.0 - 2**(-mantissa_bits))
        max_val = (2 ** max_exp) * max_mantissa

        min_exp = 1 - bias
        min_val = 2 ** (min_exp - mantissa_bits)

        x_clamped = torch.clamp(x, -max_val, max_val)

        zero_mask = x_clamped.abs() < min_val

        x_sign = torch.sign(x_clamped)
        x_abs = x_clamped.abs()
        x_abs = torch.clamp(x_abs, min=min_val)

        log_x = torch.log2(x_abs)
        exponent = torch.floor(log_x).clamp(min_exp, max_exp)

        mantissa_frac = x_abs / (2 ** exponent) - 1.0

        mantissa_levels = 1 << mantissa_bits
        mantissa_quant = torch.round(mantissa_frac * mantissa_levels) / mantissa_levels
        mantissa_quant = torch.clamp(mantissa_quant, 0, 1 - 1/mantissa_levels)

        x_quant = x_sign * (2 ** exponent) * (1.0 + mantissa_quant)
        x_quant[zero_mask] = 0.0

        return x_quant

    def forward(
        self,
        x: torch.Tensor,
        gradients: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.n_bits == 2 or self.exp_bits_per_region is None:
            threshold = x.abs().mean()
            return torch.sign(x) * (x.abs() > threshold).float() * threshold

        if gradients is None or not self.training:
            exp_bits = self.exp_bits_per_region[1]
            mantissa_bits = (self.n_bits - 1) - exp_bits
            return self._quantize_to_fp_format(x, exp_bits, mantissa_bits)

        q1, q2, q3 = self.quartile_tracker.update(gradients)
        grad_abs = gradients.abs()

        x_quant = torch.zeros_like(x)

        mask0 = grad_abs <= q1
        if mask0.any():
            exp_bits = self.exp_bits_per_region[0]
            mantissa_bits = (self.n_bits - 1) - exp_bits
            x_quant[mask0] = self._quantize_to_fp_format(x[mask0], exp_bits, mantissa_bits)

        mask1 = (grad_abs > q1) & (grad_abs <= q2)
        if mask1.any():
            exp_bits = self.exp_bits_per_region[1]
            mantissa_bits = (self.n_bits - 1) - exp_bits
            x_quant[mask1] = self._quantize_to_fp_format(x[mask1], exp_bits, mantissa_bits)

        mask2 = (grad_abs > q2) & (grad_abs <= q3)
        if mask2.any():
            exp_bits = self.exp_bits_per_region[2]
            mantissa_bits = (self.n_bits - 1) - exp_bits
            x_quant[mask2] = self._quantize_to_fp_format(x[mask2], exp_bits, mantissa_bits)

        mask3 = grad_abs > q3
        if mask3.any():
            exp_bits = self.exp_bits_per_region[3]
            mantissa_bits = (self.n_bits - 1) - exp_bits
            x_quant[mask3] = self._quantize_to_fp_format(x[mask3], exp_bits, mantissa_bits)

        return x_quant
