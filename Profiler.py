import time
import torch
import numpy as np
from typing import Dict


class PerformanceProfiler:
    """Profiler for measuring model inference performance."""

    def __init__(self):
        self.results = {}

    def _synchronize(self):
        """Synchronize CUDA device if available for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def profile_pytorch_model(self, model: torch.nn.Module, input_data: torch.Tensor,
                              model_name: str, num_runs: int = 100,
                              warmup_runs: int = 10) -> Dict[str, float]:
        """
        Profile PyTorch model inference latency.

        Args:
            model: PyTorch model to profile
            input_data: Example input tensor
            model_name: Identifier for the model in results
            num_runs: Number of timed runs
            warmup_runs: Warmup iterations to stabilize caches

        Returns:
            Dictionary with timing statistics
        """
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        model.eval()
        
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_data)
        self._synchronize()

        times = []
        for _ in range(num_runs):
            self._synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_data)
            self._synchronize()
            times.append(time.perf_counter() - start)

        times = np.array(times)
        stats = self._compute_stats(times)
        self.results[model_name] = stats
        return stats

    @staticmethod
    def _compute_stats(times: np.ndarray) -> Dict[str, float]:
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
            'throughput': float(1.0 / np.mean(times))
        }
