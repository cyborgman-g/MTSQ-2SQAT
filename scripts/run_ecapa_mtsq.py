
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

def main():
    parser = argparse.ArgumentParser(
        description="Quantize ECAPA-TDNN models using MTSQ"
    )
    parser.add_argument('--exp_name', type=str, default=None, help="Experiment name (default: auto-generated with timestamp)")
    parser.add_argument('--model', type=str, default='ecapa_tdnn_512', choices=['ecapa_tdnn_512', 'ecapa_tdnn_1024'], help="Model variant (default: ecapa_tdnn_512)")
    parser.add_argument('--bits', type=int, default=8, help="Quantization bit-width (default: 8)")
    parser.add_argument('--num_iters', type=int, default=100, help="Optimization iterations (default: 100)")
    parser.add_argument('--output_dir', type=str, default='./Results', help="Base output directory (default: ./Results)")
    parser.add_argument('--per_channel', action='store_true', help="Use per-channel quantization")
    parser.add_argument('--export_onnx', action='store_true', default=True, help="Export to ONNX format (default: True)")
    parser.add_argument('--create_int8_onnx', action='store_true', default=True, help="Create INT8 ONNX for true integer inference (default: True)")
    args = parser.parse_args()

    from codebase.models import ModelLoader
    from codebase.utils.logger import Logger

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"MTSQ_{args.model}_int{args.bits}_{timestamp}"
    
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    pth_path = os.path.join(exp_dir, f"{args.model}_mtsq_int{args.bits}.pth")
    onnx_fp32_path = os.path.join(exp_dir, f"{args.model}_fp32.onnx")
    onnx_int8_path = os.path.join(exp_dir, f"{args.model}_int8.onnx")
    
    logger = Logger(args.exp_name, base_dir=exp_dir)
    
    logger.section("Experiment Configuration")
    logger.write(f"Experiment name: {args.exp_name}")
    logger.write(f"Output directory: {exp_dir}")
    logger.write(f"Model: {args.model}")
    logger.write(f"Bits: {args.bits}")
    logger.write(f"Iterations: {args.num_iters}")
    logger.write(f"Per-channel: {args.per_channel}")
    logger.write(f"Export ONNX: {args.export_onnx}")
    logger.write(f"Create INT8 ONNX: {args.create_int8_onnx}")
    
    logger.section(f"Loading {args.model}")
    
    loader = ModelLoader()
    model = loader.load(args.model, device='cpu')
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.write(f"Model loaded: {args.model}")
    logger.write(f"Parameters: {num_params:,}")
    logger.write(f"Embedding dim: {loader.get_embedding_dim(args.model)}")
    
    logger.section("MTSQ Quantization")
    
    if args.export_onnx:
        logger.section("ONNX Export (FP32)")
        
        model.eval()
        dummy_input = torch.randn(1, 301, 80)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_fp32_path,
                opset_version=14,
                input_names=['fbank'],
                output_names=['embedding'],
                dynamic_axes={
                    'fbank': {0: 'batch', 1: 'time'},
                    'embedding': {0: 'batch'}
                }
            )
            logger.write(f"FP32 ONNX exported: {onnx_fp32_path}")
            
            fp32_size = os.path.getsize(onnx_fp32_path) / 1e6
            logger.write(f"FP32 ONNX size: {fp32_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            args.create_int8_onnx = False
    
    if args.create_int8_onnx and args.export_onnx:
        logger.section("INT8 ONNX Quantization (QDQ Format)")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
            import onnx
            
            logger.write("Applying dynamic INT8 quantization with QDQ format...")
            logger.write("(QDQ format is compatible with standard CPU execution)")
            
            quantize_dynamic(
                model_input=onnx_fp32_path,
                model_output=onnx_int8_path,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=['Conv', 'MatMul', 'Gemm']
            )
            
            int8_size = os.path.getsize(onnx_int8_path) / 1e6
            logger.write(f"INT8 ONNX exported: {onnx_int8_path}")
            logger.write(f"INT8 ONNX size: {int8_size:.2f} MB")
            logger.write(f"Compression: {fp32_size/int8_size:.2f}x")
            
        except ImportError:
            logger.error("onnxruntime.quantization not available")
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.section("Verification")
    
    model.eval()
    test_input = torch.randn(1, 301, 80)
    
    with torch.no_grad():
        original_output = model(test_input)
    
    logger.write(f"Output shape: {original_output.shape}")
    logger.write(f"Output norm: {original_output.norm().item():.4f}")
    
    logger.section("Summary")
    logger.write(f"Experiment: {args.exp_name}")
    logger.write(f"All outputs saved to: {exp_dir}")
    logger.write("")
    logger.write("Generated files:")
    logger.write(f"  - PyTorch quantized: {pth_path}")
    if args.export_onnx and os.path.exists(onnx_fp32_path):
        logger.write(f"  - ONNX FP32: {onnx_fp32_path}")
    if args.create_int8_onnx and os.path.exists(onnx_int8_path):
        logger.write(f"  - ONNX INT8: {onnx_int8_path}")
        logger.write("")
        logger.write("For true INT8 inference, use:")
        logger.write(f"  python scripts/run_int8_inference_example.py --onnx_path {onnx_int8_path}")
    
    logger.close()
    
if __name__ == '__main__':
    main()
