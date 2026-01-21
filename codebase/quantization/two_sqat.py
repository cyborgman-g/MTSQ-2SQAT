import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import copy

from .dynamic_fp8 import DynamicFP8Quantizer
from .quantizers import FakeQuantize
from .fusion import fuse_conv_bn_weights

class QuantizedConvBNReLU(nn.Module):
    def __init__(
        self,
        conv: nn.Module,
        bn: nn.Module,
        n_bits: int = 8
    ):
        super().__init__()
        self.n_bits = n_bits

        fused_weight, fused_bias = self._fuse(conv, bn)
        self.weight = nn.Parameter(fused_weight)
        self.bias = nn.Parameter(fused_bias)

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.is_conv1d = isinstance(conv, nn.Conv1d)

        self.weight_fp8_quantizer = DynamicFP8Quantizer(n_bits)
        self.weight_int8_quantizer = FakeQuantize(
            n_bits,
            symmetric=True,
            per_channel=True,
            num_channels=self.weight.shape[0]
        )
        self.act_quantizer = FakeQuantize(n_bits, symmetric=False)

        self.stage = 1

    def _fuse(self, conv: nn.Module, bn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_bias = conv.bias if conv.bias is not None else \
                    torch.zeros(conv.out_channels, device=conv.weight.device)

        return fuse_conv_bn_weights(
            conv.weight.data, conv_bias.data,
            bn.running_mean, bn.running_var,
            bn.weight.data, bn.bias.data, bn.eps
        )

    def set_stage(self, stage: int):
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.stage = stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_grad = self.weight.grad if self.weight.grad is not None else None

        if self.stage == 1:
            w_quant = self.weight_fp8_quantizer(self.weight, w_grad)
        else:
            w_quant = self.weight_int8_quantizer(self.weight)

        if self.is_conv1d:
            out = F.conv1d(x, w_quant, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(x, w_quant, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        out = F.relu(out)

        if self.stage == 2:
            out = self.act_quantizer(out)

        return out

class QuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits

        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

        self.weight_fp8_quantizer = DynamicFP8Quantizer(n_bits)
        self.weight_int8_quantizer = FakeQuantize(
            n_bits,
            symmetric=True,
            per_channel=True,
            num_channels=self.weight.shape[0]
        )
        self.act_quantizer = FakeQuantize(n_bits, symmetric=False)

        self.stage = 1

    def set_stage(self, stage: int):
        self.stage = stage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_grad = self.weight.grad if self.weight.grad is not None else None

        if self.stage == 1:
            w_quant = self.weight_fp8_quantizer(self.weight, w_grad)
        else:
            w_quant = self.weight_int8_quantizer(self.weight)

        out = F.linear(x, w_quant, self.bias)

        if self.stage == 2:
            out = self.act_quantizer(out)

        return out

class TwoSQATTrainer:
    def __init__(
        self,
        model: nn.Module,
        n_bits: int = 8,
        stage1_epochs: int = 5,
        total_epochs: int = 10
    ):
        self.original_model = model
        self.n_bits = n_bits
        self.stage1_epochs = stage1_epochs
        self.total_epochs = total_epochs

        self.model = self._prepare_model()
        self.quantized_layers: List[nn.Module] = []
        self._collect_quantized_layers()

    def _prepare_model(self) -> nn.Module:
        model = copy.deepcopy(self.original_model)
        model.eval()

        self._replace_layers(model)

        return model

    def _replace_layers(self, module: nn.Module, prefix: str = ''):
        children = list(module.named_children())

        i = 0
        while i < len(children):
            name, child = children[i]

            if isinstance(child, (nn.Conv1d, nn.Conv2d)):
                if i + 1 < len(children):
                    next_name, next_child = children[i + 1]

                    is_bn_match = (
                        (isinstance(child, nn.Conv1d) and isinstance(next_child, nn.BatchNorm1d)) or
                        (isinstance(child, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d))
                    )

                    if is_bn_match and child.out_channels == next_child.num_features:
                        fused = QuantizedConvBNReLU(child, next_child, self.n_bits)
                        setattr(module, name, fused)
                        setattr(module, next_name, nn.Identity())
                        self.quantized_layers.append(fused)
                        i += 2
                        continue

            elif isinstance(child, nn.Linear):
                quant_linear = QuantizedLinear(child, self.n_bits)
                setattr(module, name, quant_linear)
                self.quantized_layers.append(quant_linear)
                i += 1
                continue

            self._replace_layers(child, f"{prefix}.{name}" if prefix else name)
            i += 1

    def _collect_quantized_layers(self):
        for module in self.model.modules():
            if isinstance(module, (QuantizedConvBNReLU, QuantizedLinear)):
                if module not in self.quantized_layers:
                    self.quantized_layers.append(module)

    def set_stage(self, stage: int):
        for layer in self.quantized_layers:
            layer.set_stage(stage)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, List[float]]:
        self.model = self.model.to(device)
        history = {'loss': [], 'stage': [], 'lr': []}

        for epoch in range(self.total_epochs):
            if epoch < self.stage1_epochs:
                current_stage = 1
                stage_name = "FP8"
            else:
                current_stage = 2
                stage_name = "INT8"

            self.set_stage(current_stage)

            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                output = self.model(data)
                
                if output.dim() == 3 and output.size(1) == 1:
                    output = output.squeeze(1)
                
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']

            history['loss'].append(avg_loss)
            history['stage'].append(current_stage)
            history['lr'].append(current_lr)

        return history

    def get_quantized_model(self) -> nn.Module:
        return self.model

    def export_weights(self) -> Dict[str, Dict]:
        weights = {}

        for i, layer in enumerate(self.quantized_layers):
            name = f"layer_{i}"

            with torch.no_grad():
                if isinstance(layer, QuantizedConvBNReLU):
                    w_int8 = layer.weight_int8_quantizer.get_quantized(layer.weight)
                    weights[name] = {
                        'type': 'conv',
                        'weight_int8': w_int8,
                        'weight_scale': layer.weight_int8_quantizer.scale.clone(),
                        'weight_zero_point': layer.weight_int8_quantizer.zero_point.clone(),
                        'act_scale': layer.act_quantizer.scale.clone(),
                        'act_zero_point': layer.act_quantizer.zero_point.clone(),
                        'bias_fp32': layer.bias.clone()
                    }
                elif isinstance(layer, QuantizedLinear):
                    w_int8 = layer.weight_int8_quantizer.get_quantized(layer.weight)
                    weights[name] = {
                        'type': 'linear',
                        'weight_int8': w_int8,
                        'weight_scale': layer.weight_int8_quantizer.scale.clone(),
                        'weight_zero_point': layer.weight_int8_quantizer.zero_point.clone(),
                        'act_scale': layer.act_quantizer.scale.clone(),
                        'act_zero_point': layer.act_quantizer.zero_point.clone(),
                        'bias_fp32': layer.bias.clone() if layer.bias is not None else None
                    }

        return weights

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'quantized_weights': self.export_weights(),
            'config': {
                'n_bits': self.n_bits,
                'stage1_epochs': self.stage1_epochs,
                'total_epochs': self.total_epochs
            }
        }, path)
