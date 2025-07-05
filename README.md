# MTSQ — Minimal Training Static Quantization

**MTSQ** is a **data-independent post-training quantization** technique that learns optimal scale and zero-point for every layer — **without calibration data**.

---

## ✨ 1. Key Concepts

| Component                            | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| 🧪 Dual-loss Objective               | Combines reconstruction error and distribution preservation (KL-divergence) |
| 📊 Asymmetric Per-Channel Quant     | Full dynamic range without saturation                                      |
| 🔗 Activation Parameters from Next Layer | Removes sample dependency; preserves unbiasedness                     |

---

## 🗂 2. Project Structure

- *MTSQ_Utils* :- All MTSQ related code function
- *ONNX_Utils* :- ALL ONNNX related code function
- *Profiler* :- Python profiling code for measuring performance
- *main* :- Main Quantization and Compare logic
---

## ⚙️ 3. Installation
```bash
# Install required libraries
# This code need PyTorch version >2.0
pip install onnx, onnxruntime
```
---
## 📘 4. Demo: WAVLM_BASE_PLUS

```bash
python3 main.py
```

### Perfromance Results:
==== PERFORMANCE SUMMARY ====
FP32 mean latency      : 1372.57 ms
MTSQ PyTorch latency   : 1431.30 ms
ONNX (TensorRT) latency: 992.47 ms
ONNX accuracy pass rate: 100.00% | max error 0.000000

~27% Faster inference on CPU.

## 2SQAT will be updated soon
