import torch, os
from torchaudio.pipelines import WAVLM_BASE_PLUS

from Profiler import PerformanceProfiler
from MTSQ_Utils import MTSQConfig, InputQuantizer, apply_mtsq_to_model, quantize_model_weights
from ONNX_Utils import export_mtsq_to_onnx, create_optimized_session, benchmark_onnx_model, validate_onnx_accuracy

print("start")

class WavLMBasePlus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bundle = WAVLM_BASE_PLUS
        self.model = bundle.get_model()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


def generate_dummy_audio(batch: int = 12, length: int = 96000, seed: int = 42):
    """Generate random dummy audio data for benchmarking."""
    torch.manual_seed(seed)
    return torch.randn(batch, length)

def main(save_path): 
    device = torch.device('cpu')
    fp32_model = WavLMBasePlus().to(device).eval()

    test_input = generate_dummy_audio().to(device)

    profiler = PerformanceProfiler()
    fp32_stats = profiler.profile_pytorch_model(fp32_model, test_input, 'FP32')

    """
    Tweak with this config or run a grid-search for getting best configs based on models
    """
    config = MTSQConfig(bits=8,
                        symmetric=False,
                        per_channel=True,
                        power_of_two_clipping=True,
                        learning_rate=1e-3,
                        max_iterations=60,
                        patience=30)


    quant_model = apply_mtsq_to_model(fp32_model, config)

    first_params = quantize_model_weights(quant_model)
    input_quantizer = None
    if first_params is not None:
        input_quantizer = InputQuantizer(first_params[0], first_params[1], qmin=config.qmin, qmax=config.qmax)

    if config.compile_model and torch.__version__ >= '2':
        pass

    quant_stats = profiler.profile_pytorch_model(quant_model, test_input, 'MTSQ-PyTorch')

    onnx_path = os.path.join(save_path, 'wavlm_mtsq.onnx')
    export_success = export_mtsq_to_onnx(quant_model, test_input[:1], onnx_path, input_quantizer)

    if not export_success:
        print("ONNX export failed – aborting")
        return


    session = create_optimized_session(onnx_path)

    test_input_np = test_input.cpu().numpy()
    if input_quantizer is not None:
        test_input_quant = input_quantizer.quantize_input(test_input).cpu().numpy()
    else:
        test_input_quant = test_input_np

    onnx_stats = benchmark_onnx_model(session, test_input_quant)
    print(f"ONNX mean latency: {onnx_stats['mean']*1000:.2f} ms")

    val_results = validate_onnx_accuracy(quant_model, session, test_input[:2], input_quantizer)

    # Summary
    print("==== PERFORMANCE SUMMARY ====")
    print(f"FP32 mean latency      : {fp32_stats['mean']*1000:.2f} ms")
    print(f"MTSQ PyTorch latency   : {quant_stats['mean']*1000:.2f} ms | speed-up {fp32_stats['mean']/quant_stats['mean']:.2f}x")
    print(f"ONNX (TensorRT) latency: {onnx_stats['mean']*1000:.2f} ms | speed-up {fp32_stats['mean']/onnx_stats['mean']:.2f}x")
    print(f"ONNX accuracy pass rate: {val_results['pass_rate']*100:.2f}% | max error {val_results['max_error']:.6f}")

if __name__ == "__main__":
    main("./")
