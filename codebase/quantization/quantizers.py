import torch
import torch.nn as nn
from torch.autograd import Function

class STEQuantize(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        q_min: int,
        q_max: int
    ) -> torch.Tensor:
        q_val = torch.clamp(
            torch.round(input / scale) + zero_point,
            q_min,
            q_max
        )

        output = (q_val - zero_point) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None, None

class FakeQuantize(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        num_channels: int = 1
    ):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.num_channels = num_channels

        if symmetric:
            self.q_min = -(1 << (n_bits - 1))
            self.q_max = (1 << (n_bits - 1)) - 1
        else:
            self.q_min = 0
            self.q_max = (1 << n_bits) - 1

        if per_channel:
            self.register_buffer('scale', torch.ones(num_channels))
            self.register_buffer('zero_point', torch.zeros(num_channels))
        else:
            self.register_buffer('scale', torch.tensor(1.0))
            self.register_buffer('zero_point', torch.tensor(0.0))

        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_scale(x)

            if self.per_channel and x.dim() > 1:
                scale = self.scale.view(-1, *([1] * (x.dim() - 1)))
                zp = self.zero_point.view(-1, *([1] * (x.dim() - 1)))
            else:
                scale = self.scale
                zp = self.zero_point

            return STEQuantize.apply(x, scale, zp, self.q_min, self.q_max)
        else:
            if self.per_channel and x.dim() > 1:
                scale = self.scale.view(-1, *([1] * (x.dim() - 1)))
                zp = self.zero_point.view(-1, *([1] * (x.dim() - 1)))
            else:
                scale = self.scale
                zp = self.zero_point

            q_val = torch.clamp(
                torch.round(x / scale) + zp,
                self.q_min, self.q_max
            )
            return q_val.to(torch.int8)

    def _update_scale(self, x: torch.Tensor):
        with torch.no_grad():
            if self.per_channel and x.dim() > 1:
                x_flat = x.view(x.shape[0], -1)
                x_max = x_flat.abs().max(dim=1)[0]
                x_min = x_flat.min(dim=1)[0]
            else:
                x_max = x.abs().max()
                x_min = x.min()

            if self.symmetric:
                new_scale = x_max / self.q_max
                new_zp = torch.zeros_like(new_scale) if self.per_channel else torch.tensor(0.0)
            else:
                new_scale = (x_max - x_min) / (self.q_max - self.q_min)
                new_zp = self.q_min - x_min / (new_scale + 1e-8)

            new_scale = torch.clamp(new_scale, min=1e-8)

            if self.initialized:
                momentum = 0.9
                self.scale.copy_(momentum * self.scale + (1 - momentum) * new_scale)
                self.zero_point.copy_(momentum * self.zero_point + (1 - momentum) * new_zp)
            else:
                self.scale.copy_(new_scale)
                self.zero_point.copy_(new_zp)
                self.initialized = True

    def get_quantized(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.per_channel and x.dim() > 1:
                scale = self.scale.view(-1, *([1] * (x.dim() - 1)))
                zp = self.zero_point.view(-1, *([1] * (x.dim() - 1)))
            else:
                scale = self.scale
                zp = self.zero_point

            q_val = torch.clamp(
                torch.round(x / scale) + zp,
                self.q_min, self.q_max
            )
            return q_val.to(torch.int8)
