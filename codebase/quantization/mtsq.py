import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .fusion import fuse_conv_bn_eval

@dataclass
class MTSQConfig:
    bits: int = 8
    num_iters: int = 100
    lr: float = 0.01
    kl_weight: float = 1.0
    temperature: float = 1.0
    per_channel: bool = True
    symmetric: bool = True
    fuse_bn: bool = True

class MTSQQuantizer:
    def __init__(self, config: MTSQConfig):
        self.config = config

        if config.symmetric:
            self.q_min = -(2 ** (config.bits - 1))
            self.q_max = 2 ** (config.bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** config.bits - 1

    def quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        q_tensor = torch.clamp(
            torch.round(tensor / scale) + zero_point,
            self.q_min,
            self.q_max
        )
        return q_tensor

    def dequantize(
        self,
        q_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        return scale * (q_tensor - zero_point)

    def _compute_loss(
        self,
        original: torch.Tensor,
        dequantized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_frob = torch.norm(original - dequantized, p='fro')

        T = self.config.temperature

        orig_flat = original.flatten()
        deq_flat = dequantized.flatten()

        P = F.softmax(orig_flat / T, dim=0)
        Q_log = F.log_softmax(deq_flat / T, dim=0)

        loss_kl = F.kl_div(Q_log, P, reduction='batchmean')

        total_loss = loss_frob + self.config.kl_weight * loss_kl

        return total_loss, loss_frob, loss_kl

    def optimize_layer(
        self,
        weight: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = weight.device

        if self.config.per_channel:
            out_channels = weight.shape[0]
            weight_flat = weight.view(out_channels, -1)
            w_min = weight_flat.min(dim=1)[0]
            w_max = weight_flat.max(dim=1)[0]

            if self.config.symmetric:
                abs_max = torch.max(w_min.abs(), w_max.abs())
                initial_scale = abs_max / self.q_max
                initial_scale = torch.clamp(initial_scale, min=1e-8)
                initial_zero_point = torch.zeros(out_channels, device=device)
            else:
                initial_scale = (w_max - w_min) / (self.q_max - self.q_min)
                initial_scale = torch.clamp(initial_scale, min=1e-8)
                initial_zero_point = self.q_min - w_min / initial_scale

            scale_shape = [out_channels] + [1] * (weight.dim() - 1)
        else:
            w_min, w_max = weight.min(), weight.max()

            if self.config.symmetric:
                abs_max = max(abs(w_min), abs(w_max))
                initial_scale = abs_max / self.q_max
                initial_zero_point = torch.tensor(0.0, device=device)
            else:
                initial_scale = (w_max - w_min) / (self.q_max - self.q_min)
                initial_zero_point = self.q_min - w_min / initial_scale

            initial_scale = torch.clamp(initial_scale, min=1e-8)
            scale_shape = [1]

        scale = nn.Parameter(initial_scale.clone().view(*scale_shape))
        zero_point = nn.Parameter(initial_zero_point.clone().view(*scale_shape))

        optimizer = optim.SGD([scale, zero_point], lr=self.config.lr)

        for i in range(self.config.num_iters):
            optimizer.zero_grad()

            zp_rounded = torch.round(zero_point)
            zp_ste = zero_point + (zp_rounded - zero_point).detach()

            q_weight = self.quantize(weight, scale, zp_ste)
            deq_weight = self.dequantize(q_weight, scale, zp_ste)

            total_loss, loss_frob, loss_kl = self._compute_loss(weight, deq_weight)

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                scale.clamp_(min=1e-8)

        with torch.no_grad():
            final_scale = scale.detach()
            final_zp = torch.round(zero_point.detach()).clamp(self.q_min, self.q_max)

            q_weight = self.quantize(weight, final_scale, final_zp)

            if self.config.bits == 8:
                q_weight = q_weight.to(torch.int8)

        return q_weight, final_scale.squeeze(), final_zp.squeeze()

class MTSQModelQuantizer:
    def __init__(self, config: MTSQConfig):
        self.config = config
        self.quantizer = MTSQQuantizer(config)
        self.quantized_params: Dict[str, Dict] = {}

    def quantize_model(
        self,
        model: nn.Module
    ) -> nn.Module:
        model = model.eval()

        if self.config.fuse_bn:
            model = fuse_conv_bn_eval(model)

        layers_to_quantize = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                layers_to_quantize.append((name, module))

        for name, module in layers_to_quantize:
            weight = module.weight.data

            q_weight, scale, zero_point = self.quantizer.optimize_layer(weight)

            self.quantized_params[name] = {
                'quantized_weight': q_weight,
                'scale': scale,
                'zero_point': zero_point,
                'original_dtype': weight.dtype,
                'bias': module.bias.data.clone() if module.bias is not None else None
            }

        self._set_activation_scales()

        return model

    def _set_activation_scales(self):
        layer_names = list(self.quantized_params.keys())

        for i, name in enumerate(layer_names[:-1]):
            next_name = layer_names[i + 1]
            self.quantized_params[name]['act_scale'] = \
                self.quantized_params[next_name]['scale'].clone()
            self.quantized_params[name]['act_zero_point'] = \
                self.quantized_params[next_name]['zero_point'].clone()

        if layer_names:
            last_name = layer_names[-1]
            self.quantized_params[last_name]['act_scale'] = \
                self.quantized_params[last_name]['scale'].clone()
            self.quantized_params[last_name]['act_zero_point'] = \
                self.quantized_params[last_name]['zero_point'].clone()

    def save(self, path: str):
        torch.save({
            'config': self.config,
            'quantized_params': self.quantized_params
        }, path)

    def load(self, path: str):
        data = torch.load(path)
        self.config = data['config']
        self.quantized_params = data['quantized_params']

    def get_model_size_bytes(self) -> int:
        total_bytes = 0
        for name, params in self.quantized_params.items():
            q_weight = params['quantized_weight']
            total_bytes += q_weight.numel() * (self.config.bits // 8)
            if params['bias'] is not None:
                total_bytes += params['bias'].numel() * 4
            total_bytes += params['scale'].numel() * 4
            total_bytes += 4
        return total_bytes
