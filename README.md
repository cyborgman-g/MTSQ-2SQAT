# Quantization Framework

Speaker verification quantization with MTSQ and 2SQAT. [Paper Link](https://ieeexplore.ieee.org/document/10684732)

## Install
```bash
pip install -r requirements.txt
```

## MTSQ Quantization
```bash
python scripts/run_ecapa_mtsq.py --exp_name=EXP1 --model=ecapa_tdnn_512 --bits=8
```

## 2SQAT Training
```bash
python scripts/run_ecapa_2sqat.py --exp_name=EXP2 --model=ecapa_tdnn_512 --epochs=10
```

## INT8 Inference
```bash
python scripts/run_int8_inference_example.py --onnx_path=Results/EXP1/ecapa_tdnn_512_int8.onnx
```

## Python API
```python
from codebase.models import ModelLoader
from scripts.quantize_mtsq import quantize_mtsq

loader = ModelLoader()
model = loader.load('ecapa_tdnn_512', device='cpu')
quantizer = quantize_mtsq(model, 'output.pth', bits=8)
```

## INT8 Python API
```python
from codebase.inference import TrueInt8Inference

inferencer = TrueInt8Inference('model_int8.onnx')
embedding = inferencer.infer(fbank_features)
```
