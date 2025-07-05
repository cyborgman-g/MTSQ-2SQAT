import torch, copy
import torch.nn as nn
from typing import Optional
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple, Optional



@dataclass
class MTSQConfig:
    """
    Configuration class for MTSQ quantization parameters and optimization settings.

    Attributes:
        bits (int): Target quantization bit-width (default: 8)
        symmetric (bool): Whether to use symmetric quantization (default: False)
        per_channel (bool): Enable per-channel quantization for weights (default: True)
        learning_rate (float): Learning rate for scale/zero-point optimization (default: 1e-3)
        max_iterations (int): Maximum optimization iterations per layer (default: 80)
        patience (int): Early stopping patience (default: 20)
        epsilon (float): Numerical stability constant (default: 1e-8)
        scale_min (float): Minimum allowed scale value (default: 1e-6)
        power_of_two_clipping (bool): Snap scales to power-of-two for hardware efficiency (default: True)
        frobenius_weight (float): Weight for Frobenius norm loss term (default: 0.1)
        kl_weight (float): Weight for KL divergence loss term (default: 0.1)
        tolerance (float): Convergence tolerance (default: 1e-6)
        device (Optional[str]): Target device ('cuda', 'cpu', or None for auto-detect)
        compile_model (bool): Enable torch.compile for acceleration (default: True)
        use_inductor_fusion (bool): Enable Inductor INT8 fusion (default: True)
    """

    # Quantization parameters
    bits: int = 8
    symmetric: bool = False
    per_channel: bool = True

    # Optimization parameters
    learning_rate: float = 1e-3
    max_iterations: int = 80
    patience: int = 20
    epsilon: float = 1e-8
    scale_min: float = 1e-6
    tolerance: float = 1e-6

    # MTSQ-specific parameters
    frobenius_weight: float = 0.1
    kl_weight: float = 0.1
    power_of_two_clipping: bool = True

    # Runtime optimization
    device: Optional[str] = None
    compile_model: bool = True
    use_inductor_fusion: bool = True
    

    def __post_init__(self):
        self.device = 'cpu'
        
        if self.symmetric:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = 2 ** (self.bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** self.bits - 1

        # Enable Inductor optimizations for INT8 fusion
        if self.use_inductor_fusion and torch.cuda.is_available():
            torch._inductor.config.force_fuse_int_mm_with_mul = True
            torch._inductor.config.max_autotune = True

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def validate(self) -> None:
        if self.bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {self.bits}")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")

        if self.frobenius_weight < 0 or self.kl_weight < 0:
            raise ValueError("Loss weights must be non-negative")

    def summary(self) -> str:
        return f"""MTSQ Configuration Summary:
        Quantization: {self.bits}-bit {'symmetric' if self.symmetric else 'asymmetric'}
        Per-channel: {self.per_channel}
        Device: {self.device}
        Learning rate: {self.learning_rate}
        Max iterations: {self.max_iterations}
        Power-of-two clipping: {self.power_of_two_clipping}
        Torch compile: {self.compile_model}
        Inductor fusion: {self.use_inductor_fusion}
        """




class MTSQQuantizer:
    """
    Core MTSQ quantization optimizer that learns scale and zero-point parameters.

    This implementation includes several GPU-friendly optimizations:
    - Per-channel quantization for better dynamic range utilization
    - Power-of-two scale clipping for hardware-efficient division
    - Numerical stability safeguards for KL divergence computation
    - Early stopping with patience-based convergence detection
    """

    def __init__(self, config: MTSQConfig):
        self.config = config
        self.device = config.get_device()
        self.qmin = config.qmin
        self.qmax = config.qmax

    def _clip_to_power_of_two(self, scale: torch.Tensor) -> torch.Tensor:
        if not self.config.power_of_two_clipping:
            return scale
        log_scale = torch.log2(torch.clamp(scale, min=1e-8))
        rounded_log = torch.round(log_scale)
        return torch.pow(2.0, rounded_log)

    def _initialize_parameters(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.per_channel and tensor.dim() >= 2:
            channel_dim = 0
            view_shape = [-1] + [1] * (tensor.dim() - 1)

            t_min = tensor.amin(dim=list(range(1, tensor.dim())), keepdim=True)[0]
            t_max = tensor.amax(dim=list(range(1, tensor.dim())), keepdim=True)[0]
        else:
            t_min = tensor.min()
            t_max = tensor.max()
            view_shape = []

        # Handle constant tensors
        range_mask = (t_max - t_min) > self.config.epsilon

        if self.config.symmetric:
            t_abs_max = torch.maximum(torch.abs(t_min), torch.abs(t_max))
            scale = torch.where(range_mask, 
                              t_abs_max / (self.qmax - self.qmin) * 2,
                              torch.tensor(1.0, device=self.device))
            zero_point = torch.zeros_like(scale)
        else:
            scale = torch.where(range_mask,
                              (t_max - t_min) / (self.qmax - self.qmin),
                              torch.tensor(1.0, device=self.device))
            zero_point = torch.where(range_mask,
                                   self.qmin - t_min / scale,
                                   torch.tensor(0.0, device=self.device))

        scale = self._clip_to_power_of_two(scale)
        scale = scale.clone().detach().requires_grad_(True)
        zero_point = zero_point.clone().detach().requires_grad_(True)
        return scale, zero_point

    def quantize(self, tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor = torch.clamp(q_tensor, self.qmin, self.qmax)
        return q_tensor


    def dequantize(self, q_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        return scale * (q_tensor - zero_point)


    def _compute_frobenius_loss(self, original: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:
        return torch.norm(original - dequantized, p='fro')


    def _compute_kl_loss(self, original: torch.Tensor, dequantized: torch.Tensor) -> torch.Tensor:

        eps = self.config.epsilon
        orig_pos = torch.abs(original) + eps
        deq_pos = torch.abs(dequantized) + eps

        orig_sum = torch.sum(orig_pos)
        deq_sum = torch.sum(deq_pos)

        orig_norm = orig_pos / (orig_sum + eps)
        deq_norm = deq_pos / (deq_sum + eps)
        kl_div = orig_norm * torch.log((orig_norm + eps) / (deq_norm + eps))
        kl_div = torch.where(torch.isfinite(kl_div), kl_div, torch.zeros_like(kl_div))
        return torch.sum(kl_div)

    def optimize_parameters(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = tensor.to(self.device)
        scale, zero_point = self._initialize_parameters(tensor)

        optimizer = optim.Adam([scale, zero_point], lr=self.config.learning_rate)

        best_loss = float('inf')
        patience_counter = 0

        for iteration in range(self.config.max_iterations):

            optimizer.zero_grad()

            with torch.no_grad():
                scale.data = torch.clamp(scale.data, min=self.config.scale_min)
                if self.config.power_of_two_clipping:
                    scale.data = self._clip_to_power_of_two(scale.data)
                    
            q_tensor = self.quantize(tensor, scale, zero_point)
            dequantized = self.dequantize(q_tensor, scale, zero_point)
            frobenius_loss = self._compute_frobenius_loss(tensor, dequantized)
            kl_loss = self._compute_kl_loss(tensor, dequantized)
            total_loss = (self.config.frobenius_weight * frobenius_loss + self.config.kl_weight * kl_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([scale, zero_point], max_norm=1.0)
            optimizer.step()

            current_loss = total_loss.item()
            if current_loss < best_loss - self.config.tolerance:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

        with torch.no_grad():
            scale.data = torch.clamp(scale.data, min=self.config.scale_min)
            if self.config.power_of_two_clipping:
                scale.data = self._clip_to_power_of_two(scale.data)
        return scale.detach(), zero_point.detach()

    def get_quantization_error(self, tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> float:
        with torch.no_grad():
            q_tensor = self.quantize(tensor, scale, zero_point)
            dequantized = self.dequantize(q_tensor, scale, zero_point)
            error = torch.mean((tensor - dequantized) ** 2)
            return error.item()




class MTSQLayer(nn.Module):
    """
    Universal wrapper for MTSQ quantization of PyTorch layers.

    This wrapper preserves the original layer's interface while applying
    learned quantization parameters. It supports torch.compile() for
    additional performance optimization.
    """

    def __init__(self, original_layer: nn.Module, config: MTSQConfig):
        super(MTSQLayer, self).__init__()
        self.original_layer = original_layer
        self.config = config
        self.quantizer = MTSQQuantizer(config)
        
        if isinstance(original_layer, nn.Linear):
            num_output = original_layer.out_features
        elif isinstance(original_layer, (nn.Conv1d, nn.Conv2d)):
            num_output = original_layer.out_channels
        else:
            num_output = 1
        
        self.register_buffer('original_weight', original_layer.weight.data.clone())
        self.register_buffer('weight_scale', torch.ones(num_output))
        self.register_buffer('weight_zero_point', torch.zeros(num_output))
        
        self.is_quantized = False

    def apply_quantization(self):
        if self.is_quantized:
            return

        weight_scale, weight_zero_point = self.quantizer.optimize_parameters(self.original_layer.weight.data)
    
        if weight_scale.shape != self.weight_scale.shape:
            self.weight_scale = torch.ones_like(weight_scale)
            self.weight_zero_point = torch.zeros_like(weight_zero_point)
        
        self.weight_scale.copy_(weight_scale)
        self.weight_zero_point.copy_(weight_zero_point)

        q_weight = self.quantizer.quantize(
            self.original_layer.weight.data, 
            self.weight_scale, 
            self.weight_zero_point
        )

        dequantized_weight = self.quantizer.dequantize(
            q_weight, 
            self.weight_scale, 
            self.weight_zero_point
        )

        self.original_layer.weight.data.copy_(dequantized_weight)
        self.is_quantized = True

        # Log quantization statistics
        error = self.get_quantization_error()
        print(f"Quantization applied with MSE: {error:.6f}")

    def get_quantization_error(self) -> float:
        if not self.is_quantized or self.original_weight is None:
            return 0.0
        error = torch.mean((self.original_weight - self.original_layer.weight.data) ** 2)
        return error.item()

    def get_quantization_stats(self) -> dict:
        if not self.is_quantized:
            return {"error": "Layer not quantized"}

        stats = {
            "quantized": True,
            "mse_error": self.get_quantization_error(),
            "scale_range": (self.weight_scale.min().item(), self.weight_scale.max().item()),
            "zero_point_range": (self.weight_zero_point.min().item(), self.weight_zero_point.max().item()),
            "per_channel": self.config.per_channel and self.weight_scale.numel() > 1,
            "weight_shape": tuple(self.original_layer.weight.shape) if hasattr(self.original_layer, 'weight') else None
        }

        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_layer, name)

    def __repr__(self) -> str:
        status = "quantized" if self.is_quantized else "not quantized"
        return f"MTSQLayer({self.original_layer}, {status})"


class InputQuantizer:
    """
    Handles input quantization using first layer's quantization parameters.

    This ensures consistency between input quantization and the model's
    internal quantization scheme as specified in the MTSQ paper.
    """

    def __init__(self, first_layer_scale: torch.Tensor, 
                 first_layer_zero_point: torch.Tensor,
                 qmin: int = 0, qmax: int = 255):
        """
        Initialize input quantizer.

        Args:
            first_layer_scale: Scale parameter from first quantized layer
            first_layer_zero_point: Zero-point parameter from first quantized layer
            qmin: Minimum quantized value
            qmax: Maximum quantized value
        """
        self.scale = first_layer_scale.cpu()
        self.zero_point = first_layer_zero_point.cpu()
        self.qmin = qmin
        self.qmax = qmax

    def quantize_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        device = input_tensor.device
        scale = self.scale.to(device)
        zero_point = self.zero_point.to(device)
        
        q_input = torch.round(input_tensor / scale + zero_point)
        q_input = torch.clamp(q_input, self.qmin, self.qmax)

        return scale * (q_input - zero_point)

    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scale.clone(), self.zero_point.clone()



def apply_mtsq_to_model(model: nn.Module, 
                       config: MTSQConfig,
                       target_layers: tuple = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) -> nn.Module:
    """
    Apply MTSQ quantization to all specified layer types in a model.

    This function recursively traverses the model and replaces target layers
    with MTSQ-quantized equivalents.

    Args:
        model: PyTorch model to quantize
        config: MTSQ configuration
        target_layers: Tuple of layer types to quantize

    Returns:
        Model with MTSQ layers applied (deep copy of original)
    """
    def replace_layers(module: nn.Module, name: str = "") -> None:
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child_module, target_layers):
                mtsq_layer = MTSQLayer(child_module, config)
                setattr(module, child_name, mtsq_layer)
            else:
                replace_layers(child_module, full_name)


    quantized_model = copy.deepcopy(model)
    replace_layers(quantized_model)

    print(f"Applied MTSQ to model with {len(list(quantized_model.modules()))} total modules")
    return quantized_model


def quantize_model_weights(model: nn.Module) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply quantization to all MTSQ layers in a model and return first layer parameters.

    Args:
        model: Model containing MTSQ layers

    Returns:
        Tuple of (scale, zero_point) from first quantized layer, or None if no layers found
    """
    mtsq_layers = []
    first_layer_params = None

    def find_mtsq_layers(module: nn.Module) -> None:
        nonlocal first_layer_params

        for child in module.children():
            if isinstance(child, MTSQLayer):
                mtsq_layers.append(child)
                
                if not child.is_quantized:
                    child.apply_quantization()

                if first_layer_params is None and child.weight_scale is not None:
                    first_layer_params = (
                        child.weight_scale.clone(),
                        child.weight_zero_point.clone()
                    )
            else:
                find_mtsq_layers(child)

    find_mtsq_layers(model)
    return first_layer_params


def get_model_quantization_stats(model: nn.Module) -> dict:
    """
    Get comprehensive quantization statistics for a model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary containing aggregated quantization statistics
    """
    mtsq_layers = []
    total_params = 0
    quantized_params = 0
    total_error = 0.0

    def collect_stats(module: nn.Module) -> None:
        nonlocal total_params, quantized_params, total_error

        for child in module.children():
            if isinstance(child, MTSQLayer):
                mtsq_layers.append(child)
                if hasattr(child.original_layer, 'weight') and child.original_layer.weight is not None:
                    layer_params = child.original_layer.weight.numel()
                    total_params += layer_params

                    if child.is_quantized:
                        quantized_params += layer_params
                        total_error += child.get_quantization_error() * layer_params
            else:
                collect_stats(child)

    collect_stats(model)

    stats = {
        "total_mtsq_layers": len(mtsq_layers),
        "total_parameters": total_params,
        "quantized_parameters": quantized_params,
        "quantization_ratio": quantized_params / total_params if total_params > 0 else 0.0,
        "average_mse_error": total_error / quantized_params if quantized_params > 0 else 0.0,
        "memory_reduction_ratio": 0.75 if quantized_params > 0 else 0.0  # 8-bit vs 32-bit
    }

    return stats
