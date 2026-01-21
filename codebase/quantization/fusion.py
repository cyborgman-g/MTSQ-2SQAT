import torch
import torch.nn as nn
from typing import Tuple
import copy

def fuse_conv_bn_weights(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_gamma: torch.Tensor,
    bn_beta: torch.Tensor,
    bn_eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    w_bn = bn_gamma / torch.sqrt(bn_var + bn_eps)

    if conv_weight.dim() == 3:
        w_bn_reshaped = w_bn.view(-1, 1, 1)
    elif conv_weight.dim() == 4:
        w_bn_reshaped = w_bn.view(-1, 1, 1, 1)
    else:
        w_bn_reshaped = w_bn.view(-1, 1)

    fused_weight = conv_weight * w_bn_reshaped
    fused_bias = w_bn * (conv_bias - bn_mean) + bn_beta

    return fused_weight, fused_bias

def fuse_conv_bn_module(
    conv: nn.Module,
    bn: nn.Module
) -> nn.Module:
    assert not bn.training, "BN must be in eval mode for fusion"

    conv_weight = conv.weight.data
    conv_bias = conv.bias.data if conv.bias is not None else \
                torch.zeros(conv.out_channels, device=conv_weight.device)

    fused_weight, fused_bias = fuse_conv_bn_weights(
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_gamma=bn.weight,
        bn_beta=bn.bias,
        bn_eps=bn.eps
    )

    if isinstance(conv, nn.Conv1d):
        fused_conv = nn.Conv1d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
        )
    elif isinstance(conv, nn.Conv2d):
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
        )
    else:
        raise ValueError(f"Unsupported conv type: {type(conv)}")

    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias

    return fused_conv

def _get_parent_module(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

def fuse_conv_bn_eval(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    model.eval()

    fusions = []

    module_list = list(model.named_modules())
    module_dict = dict(module_list)

    for i, (name, module) in enumerate(module_list[:-1]):
        next_name, next_module = module_list[i + 1]

        is_conv1d = isinstance(module, nn.Conv1d)
        is_conv2d = isinstance(module, nn.Conv2d)
        is_bn1d = isinstance(next_module, nn.BatchNorm1d)
        is_bn2d = isinstance(next_module, nn.BatchNorm2d)

        can_fuse = (is_conv1d and is_bn1d) or (is_conv2d and is_bn2d)

        if can_fuse:
            if module.out_channels == next_module.num_features:
                fusions.append((name, next_name))

    for conv_name, bn_name in fusions:
        conv = module_dict[conv_name]
        bn = module_dict[bn_name]

        fused = fuse_conv_bn_module(conv, bn)

        parent, child_name = _get_parent_module(model, conv_name)
        setattr(parent, child_name, fused)

        bn_parent, bn_child_name = _get_parent_module(model, bn_name)
        setattr(bn_parent, bn_child_name, nn.Identity())

    return model
