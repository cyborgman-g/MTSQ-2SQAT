import onnx
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional
from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantFormat,
            QuantType
        )


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 32000),
    opset_version: int = 14,
    dynamic_batch: bool = True
) -> str:
    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'audio': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['audio'],
        output_names=['embedding'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path

def quantize_onnx_static(
    onnx_model_path: str,
    output_path: str,
    calibration_data,
    per_channel: bool = True
) -> str:

    class SimpleCalibrationReader(CalibrationDataReader):
        def __init__(self, data):
            if hasattr(data, '__iter__') and hasattr(data, '__len__'):
                self.data_iter = iter(data)
                self.is_loader = True
            else:
                self.data = data
                self.is_loader = False
                self.idx = 0

            self.count = 0
            self.max_samples = 100

        def get_next(self):
            if self.count >= self.max_samples:
                return None

            try:
                if self.is_loader:
                    audio, _ = next(self.data_iter)
                    self.count += 1
                    return {'audio': audio.numpy().astype(np.float32)}
                else:
                    if self.idx >= len(self.data):
                        return None
                    sample = self.data[self.idx:self.idx+1]
                    self.idx += 1
                    self.count += 1
                    return {'audio': sample.astype(np.float32)}
            except StopIteration:
                return None

    calibration_reader = SimpleCalibrationReader(calibration_data)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=onnx_model_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        optimize_model=True
    )

    return output_path
