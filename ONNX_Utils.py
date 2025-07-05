import onnx
import torch
import numpy
import torch.nn as nn
import onnxruntime as ort
from typing import Optional, List, Dict, Any, Tuple

from MTSQ_Utils import InputQuantizer


class ONNXExportConfig:
    """Configuration for ONNX export options."""
    def __init__(self,
                 opset_version: int = 14,
                 dynamic_axes: bool = True,
                 optimization_level: str = 'all',
                 use_external_data: bool = False):
        """
        Initialize ONNX export configuration.

        Args:
            opset_version: ONNX opset version (14+ recommended for quantization)
            dynamic_axes: Whether to export with dynamic batch dimensions
            optimization_level: ONNX graph optimization level
            use_external_data: Whether to store large tensors externally
        """
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        self.optimization_level = optimization_level
        self.use_external_data = use_external_data


def export_mtsq_to_onnx(model: nn.Module,
                        input_sample: torch.Tensor,
                        output_path: str,
                        input_quantizer: Optional[InputQuantizer] = None,
                        config: Optional[ONNXExportConfig] = None,
                        input_names: List[str] = None,
                        output_names: List[str] = None) -> bool:
    """
    Export MTSQ-quantized model to ONNX format.

    This function handles the export process with proper quantization-aware
    settings and creates QDQ (Quantize-Dequantize) patterns that TensorRT
    can optimize into fused INT8 kernels.

    Args:
        model: Quantized PyTorch model
        input_sample: Sample input tensor for tracing
        output_path: Path to save ONNX model
        input_quantizer: Optional input quantizer for end-to-end quantization
        config: Export configuration options
        input_names: Names for input tensors
        output_names: Names for output tensors

    Returns:
        True if export succeeded, False otherwise
    """
    if config is None:
        config = ONNXExportConfig()

    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    try:
        model.eval()
        export_input = input_sample
        if input_quantizer is not None:
            export_input = input_quantizer.quantize_input(input_sample)
            
        dynamic_axes_dict = {}
        if config.dynamic_axes:
            for name in input_names:
                dynamic_axes_dict[name] = {0: 'batch_size'}
            for name in output_names:
                dynamic_axes_dict[name] = {0: 'batch_size'}

        torch.onnx.export(
            model,
            export_input,
            output_path,
            export_params=True,
            opset_version=config.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict if config.dynamic_axes else None,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Log model information
        input_info = [(input.name, input.type.tensor_type.shape) for input in onnx_model.graph.input]
        output_info = [(output.name, output.type.tensor_type.shape) for output in onnx_model.graph.output]
        return True

    except Exception as e:
        return False

def create_optimized_session(onnx_path: str,
                             providers: Optional[List[str]] = None,
                             provider_options: Optional[List[Dict]] = None,
                             session_options: Optional[ort.SessionOptions] = None) -> ort.InferenceSession:
    """
    Create an optimized ONNX Runtime session with hardware-specific providers.

    This function configures ONNX Runtime for maximum performance by selecting
    the best available execution providers in priority order.

    Args:
        onnx_path: Path to ONNX model file
        providers: List of execution providers in priority order
        provider_options: Configuration options for each provider
        session_options: Additional session configuration

    Returns:
        Configured ONNX Runtime inference session
    """
    if session_options is None:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = torch.get_num_threads()
        session_options.inter_op_num_threads = 1

        # Enable memory optimizations
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_profiling = False  # Disable for production

    if providers is None:
        providers, provider_options = get_optimal_providers()

    try:
        session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options
        )

        # Log active providers
        active_providers = session.get_providers()
        print(f"Created ONNX session with providers: {active_providers}")
        return session

    except Exception as e:
        fallback_session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )
        return fallback_session

def get_optimal_providers() -> Tuple[List[str], List[Dict]]:
    """
    Determine the optimal execution providers for the current hardware.

    Returns:
        Tuple of (providers_list, provider_options_list)
    """
    providers = []
    provider_options = []

    available_providers = ort.get_available_providers()

    if 'TensorrtExecutionProvider' in available_providers:
        providers.append('TensorrtExecutionProvider')
        provider_options.append({
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_int8_enable': True,
            'trt_int8_use_native_calibration_table': False,
            'trt_max_workspace_size': 1 << 30,  # 1GB
            'trt_fp16_enable': False,  # Prefer INT8 over FP16
            'trt_dla_enable': False,
            'trt_dump_ep_context_model': False,
        })
        print("TensorRT execution provider available - enabling INT8 optimizations")


    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
        provider_options.append({
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
        print("CUDA execution provider available")

    providers.append('CPUExecutionProvider')
    provider_options.append({
        'intra_op_num_threads': torch.get_num_threads(),
        'inter_op_num_threads': 1,
    })

    return providers, provider_options

def benchmark_onnx_model(session: ort.InferenceSession,
                        input_data: numpy.ndarray,
                        num_runs: int = 100,
                        warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark ONNX model inference performance.

    Args:
        session: ONNX Runtime inference session
        input_data: Input data for benchmarking
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary containing performance statistics
    """
    import time

    input_name = session.get_inputs()[0].name
    input_dict = {input_name: input_data}

    # Warmup runs
    for _ in range(warmup_runs):
        _ = session.run(None, input_dict)

    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, input_dict)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = numpy.array(times)

    stats = {
        'mean': numpy.mean(times),
        'std': numpy.std(times),
        'min': numpy.min(times),
        'max': numpy.max(times),
        'median': numpy.median(times),
        'p95': numpy.percentile(times, 95),
        'p99': numpy.percentile(times, 99),
        'throughput': 1.0 / numpy.mean(times)  # inferences per second
    }

    return stats

def validate_onnx_accuracy(pytorch_model: nn.Module,
                          onnx_session: ort.InferenceSession,
                          test_inputs: torch.Tensor,
                          input_quantizer: Optional[InputQuantizer] = None,
                          rtol: float = 1e-3,
                          atol: float = 1e-3) -> Dict[str, Any]:
    """
    Validate accuracy between PyTorch and ONNX models.

    Args:
        pytorch_model: Original PyTorch model
        onnx_session: ONNX Runtime session
        test_inputs: Test input tensors
        input_quantizer: Optional input quantizer
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary containing accuracy validation results
    """
    pytorch_model.eval()

    results = {
        'total_samples': test_inputs.shape[0],
        'passed_samples': 0,
        'max_error': 0.0,
        'mean_error': 0.0,
        'errors': []
    }

    input_name = onnx_session.get_inputs()[0].name

    with torch.no_grad():
        for i, input_tensor in enumerate(test_inputs):
            pytorch_input = input_tensor.unsqueeze(0)
            if input_quantizer:
                pytorch_input = input_quantizer.quantize_input(pytorch_input)

            pytorch_output = pytorch_model(pytorch_input)
            
            onnx_input = pytorch_input.cpu().numpy()
            onnx_output = onnx_session.run(None, {input_name: onnx_input})[0]

            pytorch_output_np = pytorch_output.cpu().numpy()

            try:
                numpy.testing.assert_allclose(pytorch_output_np, onnx_output, rtol=rtol, atol=atol)
                results['passed_samples'] += 1
                error = 0.0
            except AssertionError:
                error = numpy.max(numpy.abs(pytorch_output_np - onnx_output))

            results['errors'].append(error)
            results['max_error'] = max(results['max_error'], error)

    results['mean_error'] = numpy.mean(results['errors'])
    results['pass_rate'] = results['passed_samples'] / results['total_samples']

    print(f"Accuracy validation: {results['passed_samples']}/{results['total_samples']} ", f"samples passed ({results['pass_rate']:.2%})")
    print(f"Max error: {results['max_error']:.6f}, Mean error: {results['mean_error']:.6f}")

    return results


def optimize_onnx_model(input_path: str, output_path: str) -> bool:
    """
    Apply ONNX model optimizations for better inference performance.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model

    Returns:
        True if optimization succeeded, False otherwise
    """
    try:
        from onnxruntime.tools import optimizer

        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=None,
            opt_level=99  # All optimizations
        )

        # Save optimized model
        optimized_model.save_model_to_file(output_path)
        print(f"Optimized ONNX model saved to {output_path}")
        return True

    except ImportError:
        print("ONNX optimizer not available - skipping model optimization")
        return False
    except Exception as e:
        print(f"Failed to optimize ONNX model: {str(e)}")
        return False
