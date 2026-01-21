import time
import numpy as np
import onnxruntime as ort
from typing import List, Optional, Dict



class ONNXInferenceRunner:
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):

        self.ort = ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        if providers is None:
            providers = self._get_optimal_providers()

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _get_optimal_providers(self) -> List[str]:
        available = self.ort.get_available_providers()

        providers = []

        if 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')

        providers.append('CPUExecutionProvider')

        return providers

    def infer(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: audio}
        )

        return outputs[0]

    def benchmark(
        self,
        input_shape: tuple = (1, 32000),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(warmup_runs):
            self.infer(dummy_input)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p99_ms': np.percentile(times, 99)
        }
