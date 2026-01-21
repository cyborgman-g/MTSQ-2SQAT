
import os
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
from typing import Dict, List, Optional, Union, Callable

import torch
import torch.nn as nn

class TrueInt8Inference:
    
    def __init__(
        self,
        onnx_int8_path: str,
        providers: Optional[List[str]] = None,
        enable_profiling: bool = False
    ):
        
        self.ort = ort
        self.model_path = onnx_int8_path
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        
        if enable_profiling:
            sess_options.enable_profiling = True
        
        if providers is None:
            providers = self._get_optimal_providers()
        
        self.session = ort.InferenceSession(
            onnx_int8_path,
            sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        self._fbank_extractor = None
    
    def _get_optimal_providers(self) -> List[str]:
        available = self.ort.get_available_providers()
        providers = []
        
        if 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        return providers
    
    @classmethod
    def from_pytorch_model(
        cls,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        save_dir: str,
        model_name: str = "model",
        feature_extractor: Optional[Callable] = None,
        num_calibration_samples: int = 100,
        per_channel: bool = False
    ) -> "TrueInt8Inference":
        import onnx
        from onnxruntime.quantization import (
            quantize_static, 
            CalibrationDataReader,
            QuantFormat, 
            QuantType
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        fp32_path = os.path.join(save_dir, f"{model_name}_fp32.onnx")
        int8_path = os.path.join(save_dir, f"{model_name}_int8.onnx")
        
        model = model.cpu().eval()
        
        sample_batch = next(iter(calibration_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch
        
        if feature_extractor is not None:
            sample_input = feature_extractor(sample_input)
        
        if sample_input.dim() == 2:
            sample_input = sample_input.unsqueeze(0)
        
        dummy_input = torch.randn_like(sample_input[:1])
        
        torch.onnx.export(
            model,
            dummy_input,
            fp32_path,
            opset_version=14,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch', 1: 'time'},
                'output': {0: 'batch'}
            },
            do_constant_folding=True
        )
        
        onnx_model = onnx.load(fp32_path)
        onnx.checker.check_model(onnx_model)
        
        class CalibDataReader(CalibrationDataReader):
            def __init__(self, loader, feature_fn, max_samples):
                self.loader = iter(loader)
                self.feature_fn = feature_fn
                self.count = 0
                self.max_samples = max_samples
            
            def get_next(self):
                if self.count >= self.max_samples:
                    return None
                try:
                    batch = next(self.loader)
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch
                    
                    if self.feature_fn is not None:
                        data = self.feature_fn(data)
                    
                    if data.dim() == 2:
                        data = data.unsqueeze(0)
                    
                    self.count += 1
                    return {'input': data.numpy().astype(np.float32)}
                except StopIteration:
                    return None
        
        calib_reader = CalibDataReader(
            calibration_loader, 
            feature_extractor,
            num_calibration_samples
        )
        
        quantize_static(
            model_input=fp32_path,
            model_output=int8_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            optimize_model=True
        )
        
        return cls(int8_path)
    
    @property
    def fbank_extractor(self):
        if self._fbank_extractor is None:
            import torchaudio
            
            class FbankExtractor:
                def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    
                    fbank = torchaudio.compliance.kaldi.fbank(
                        waveform,
                        num_mel_bins=80,
                        sample_frequency=16000,
                        frame_length=25,
                        frame_shift=10
                    )
                    
                    if fbank.dim() == 2:
                        fbank = fbank.unsqueeze(0)
                    
                    return fbank
            
            self._fbank_extractor = FbankExtractor()
        
        return self._fbank_extractor
    
    def infer(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        extract_features: bool = False
    ) -> np.ndarray:
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu()
            
            if extract_features:
                input_data = self.fbank_extractor(input_data)
            
            input_data = input_data.numpy()
        
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :]
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        return outputs[0]
    
    def infer_batch(
        self,
        inputs: List[Union[np.ndarray, torch.Tensor]],
        extract_features: bool = False
    ) -> List[np.ndarray]:
        return [self.infer(x, extract_features) for x in inputs]
    
    def benchmark(
        self,
        input_shape: tuple = (1, 301, 80),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        for _ in range(warmup_runs):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'provider': self.session.get_providers()[0]
        }
    
    def get_model_info(self) -> Dict[str, any]:
        return {
            'path': self.model_path,
            'input_name': self.input_name,
            'input_shape': self.input_shape,
            'output_name': self.output_name,
            'providers': self.session.get_providers(),
            'size_mb': os.path.getsize(self.model_path) / 1e6
        }
    
    def compare_with_fp32(
        self,
        fp32_onnx_path: str,
        test_inputs: List[np.ndarray]
    ) -> Dict[str, float]:
        import onnxruntime as ort
        
        fp32_session = ort.InferenceSession(fp32_onnx_path)
        fp32_input_name = fp32_session.get_inputs()[0].name
        
        cosine_sims = []
        mse_errors = []
        
        for inp in test_inputs:
            if inp.ndim == 2:
                inp = inp[np.newaxis, :]
            
            fp32_out = fp32_session.run(None, {fp32_input_name: inp})[0]
            int8_out = self.infer(inp)
            
            fp32_flat = fp32_out.flatten()
            int8_flat = int8_out.flatten()
            cosine = np.dot(fp32_flat, int8_flat) / (
                np.linalg.norm(fp32_flat) * np.linalg.norm(int8_flat) + 1e-8
            )
            cosine_sims.append(cosine)
            
            mse = np.mean((fp32_out - int8_out) ** 2)
            mse_errors.append(mse)
        
        return {
            'cosine_similarity_mean': np.mean(cosine_sims),
            'cosine_similarity_std': np.std(cosine_sims),
            'mse_mean': np.mean(mse_errors),
            'mse_std': np.std(mse_errors)
        }
