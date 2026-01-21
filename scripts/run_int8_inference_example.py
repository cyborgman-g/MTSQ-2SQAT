
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(
        description="True INT8 Inference Example using ONNX Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_int8_inference_example.py --exp_name test1 --model ecapa_tdnn_512  

  python run_int8_inference_example.py --exp_name test1 --onnx_path ./Results/exp/model_int8.onnx

Note: If you have a .pth file from MTSQ/2SQAT, you need to first export it to ONNX.
      Use run_ecapa_mtsq.py or run_ecapa_2sqat.py with --export_onnx flag.
"""
    )
    parser.add_argument('--exp_name', type=str, default=None, help="Experiment name (default: auto-generated)")
    parser.add_argument('--model', type=str, default='ecapa_tdnn_512', choices=['ecapa_tdnn_512', 'ecapa_tdnn_1024'], help="Model to quantize if creating new (default: ecapa_tdnn_512)")
    parser.add_argument('--onnx_path', type=str, default=None, help="Path to existing INT8 ONNX model (.onnx file, NOT .pth!)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Config file path")
    parser.add_argument('--output_dir', type=str, default='./Results', help="Base output directory")
    parser.add_argument('--num_calibration', type=int, default=100, help="Number of calibration samples (default: 100)")
    args = parser.parse_args()

    from codebase.inference import TrueInt8Inference
    from codebase.utils.logger import Logger
    
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"INT8_Inference_{timestamp}"
    
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = Logger(args.exp_name, base_dir=exp_dir)
    
    logger.section("Experiment Configuration")
    logger.write(f"Experiment name: {args.exp_name}")
    logger.write(f"Output directory: {exp_dir}")
    
    if args.onnx_path is not None:
        if args.onnx_path.endswith('.pth'):
            logger.error("ERROR: You passed a .pth file, but TrueInt8Inference requires an ONNX file (.onnx)")
            logger.error("")
            logger.error("To create an INT8 ONNX file, run:")
            logger.error("  python scripts/run_ecapa_mtsq.py --exp_name my_exp --model ecapa_tdnn_512 --export_onnx")
            logger.error("")
            logger.error("Then use the generated .onnx file:")
            logger.error("  python scripts/run_int8_inference_example.py --onnx_path ./Results/my_exp/ecapa_tdnn_512_int8.onnx")
            logger.close()
            return
        
        if not os.path.exists(args.onnx_path):
            logger.error(f"ONNX file not found: {args.onnx_path}")
            logger.close()
            return
    
    if args.onnx_path is None:
        logger.section("Creating INT8 ONNX Model from PyTorch")
        
        from codebase.models import ModelLoader
        from codebase.models.onnx_exportable import FbankExtractor
        from torch.utils.data import DataLoader
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        logger.write(f"Loading {args.model}...")
        loader = ModelLoader()
        model = loader.load(args.model, device='cpu')
        
        train_path = config.get('data', {}).get('voxceleb2_train')
        
        if train_path and Path(train_path).exists():
            logger.write(f"Creating calibration loader from {train_path}...")
            
            from codebase.data.voxceleb import VoxCelebDataset
            
            dataset = VoxCelebDataset(
                audios_path=train_path,
                duration=3.0,
                num_classes=10,
                augment=False
            )
            
            calib_loader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            fbank_extractor = FbankExtractor()
            
            def feature_fn(audio):
                return fbank_extractor(audio)
            
            logger.write("Creating INT8 ONNX model with calibration...")
            inferencer = TrueInt8Inference.from_pytorch_model(
                model=model,
                calibration_loader=calib_loader,
                save_dir=exp_dir,
                model_name=args.model,
                feature_extractor=feature_fn,
                num_calibration_samples=args.num_calibration
            )
            
        else:
            logger.write("No calibration data available. Using random calibration...")
            
            class DummyCalibLoader:
                def __iter__(self):
                    for _ in range(100):
                        yield torch.randn(1, 301, 80)
            
            inferencer = TrueInt8Inference.from_pytorch_model(
                model=model,
                calibration_loader=DummyCalibLoader(),
                save_dir=exp_dir,
                model_name=args.model,
                feature_extractor=None,
                num_calibration_samples=50
            )
        
        args.onnx_path = os.path.join(exp_dir, f"{args.model}_int8.onnx")
        logger.write(f"INT8 ONNX model created: {args.onnx_path}")
    
    else:
        logger.section("Loading Existing INT8 ONNX Model")
        logger.write(f"Loading: {args.onnx_path}")
        
        inferencer = TrueInt8Inference(args.onnx_path)
    
    logger.section("Model Information")
    
    info = inferencer.get_model_info()
    for key, value in info.items():
        logger.write(f"  {key}: {value}")
    
    logger.section("Running Inference Example")
    
    dummy_fbank = np.random.randn(1, 301, 80).astype(np.float32)
    
    embedding = inferencer.infer(dummy_fbank)
    
    logger.write(f"Input shape: {dummy_fbank.shape}")
    logger.write(f"Output shape: {embedding.shape}")
    logger.write(f"Output norm: {np.linalg.norm(embedding):.4f}")
    logger.write(f"Output sample: {embedding.flatten()[:5]}...")
    
    logger.section("Benchmarking INT8 Inference")
    
    stats = inferencer.benchmark(num_runs=100, warmup_runs=10)
    
    logger.write(f"Execution Provider: {stats['provider']}")
    logger.write(f"Mean Latency: {stats['mean_ms']:.2f} ms")
    logger.write(f"Std Dev: {stats['std_ms']:.2f} ms")
    logger.write(f"Min: {stats['min_ms']:.2f} ms")
    logger.write(f"Max: {stats['max_ms']:.2f} ms")
    logger.write(f"P50: {stats['p50_ms']:.2f} ms")
    logger.write(f"P95: {stats['p95_ms']:.2f} ms")
    logger.write(f"P99: {stats['p99_ms']:.2f} ms")
    
    fp32_path = args.onnx_path.replace('_int8.onnx', '_fp32.onnx')
    
    if os.path.exists(fp32_path):
        logger.section("Comparing INT8 vs FP32 Accuracy")
        
        test_inputs = [
            np.random.randn(1, 301, 80).astype(np.float32)
            for _ in range(20)
        ]
        
        comparison = inferencer.compare_with_fp32(fp32_path, test_inputs)
        
        logger.write(f"Cosine Similarity: {comparison['cosine_similarity_mean']:.4f} ± {comparison['cosine_similarity_std']:.4f}")
        logger.write(f"MSE: {comparison['mse_mean']:.6f} ± {comparison['mse_std']:.6f}")
    
    logger.section("Summary")
    logger.write(f"Experiment: {args.exp_name}")
    logger.write(f"All outputs saved to: {exp_dir}")
    logger.write("")
    logger.write("Usage Example:")
    logger.write("```python")
    logger.write("from codebase.inference import TrueInt8Inference")
    logger.write("")
    logger.write(f"inferencer = TrueInt8Inference('{args.onnx_path}')")
    logger.write("embedding = inferencer.infer(fbank_features)")
    logger.write("```")
    
    logger.close()
    
if __name__ == '__main__':
    main()
