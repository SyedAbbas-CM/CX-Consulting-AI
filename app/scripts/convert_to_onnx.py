#!/usr/bin/env python3
"""
Convert Embedding Model to ONNX

This script converts a Hugging Face model to ONNX format for faster inference.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cx_consulting_ai.convert_to_onnx")


def convert_to_onnx(
    model_name: str = None, output_path: str = None, quantize: bool = False
):
    """
    Convert a Hugging Face model to ONNX format.

    Args:
        model_name: Name or path of the model to convert
        output_path: Path to save the ONNX model
        quantize: Whether to quantize the model for smaller size
    """
    try:
        # Check if transformers is installed
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Please install it: pip install transformers"
            )

        # Check if torch is installed
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch package not installed. Please install it: pip install torch"
            )

        # Check if onnx is installed
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "ONNX package not installed. Please install it: pip install onnx onnxruntime"
            )

        # Use default model if not provided
        if not model_name:
            model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")

        # Use default output path if not provided
        if not output_path:
            output_path = os.getenv("ONNX_MODEL_PATH", "models/bge-small-en-v1.5.onnx")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Converting {model_name} to ONNX format...")

        # Load model and tokenizer
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Set model to evaluation mode
        model.eval()

        # Generate dummy inputs
        dummy_inputs = tokenizer(
            "This is a test sentence",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Convert to ONNX
        import onnxruntime
        from torch.onnx import export

        logger.info(f"Exporting model to {output_path}...")

        # Export the model
        with torch.no_grad():
            export(
                model,
                (
                    dummy_inputs["input_ids"],
                    dummy_inputs["attention_mask"],
                    None,  # token_type_ids
                ),
                output_path,
                opset_version=12,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                    "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                    "pooler_output": {0: "batch_size"},
                },
            )

        logger.info("ONNX export completed")

        # Verify the model
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification successful")

        # Quantize if requested
        if quantize:
            try:
                from onnxruntime.quantization import quantize_dynamic

                # Quantize the model
                logger.info("Quantizing ONNX model...")
                quantized_path = output_path.replace(".onnx", "_quantized.onnx")
                quantize_dynamic(
                    output_path,
                    quantized_path,
                    weight_type=onnx.TensorProto.FLOAT16,
                    optimize_model=True,
                )

                logger.info(f"Quantized model saved to {quantized_path}")
            except Exception as e:
                logger.error(f"Error quantizing model: {str(e)}")

        # Test the model
        logger.info("Testing ONNX inference...")

        try:
            # Create ONNX session
            sess_options = onnxruntime.SessionOptions()
            onnx_session = onnxruntime.InferenceSession(
                output_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Run inference
            onnx_inputs = {
                "input_ids": dummy_inputs["input_ids"].numpy(),
                "attention_mask": dummy_inputs["attention_mask"].numpy(),
                "token_type_ids": dummy_inputs.get(
                    "token_type_ids", torch.zeros_like(dummy_inputs["input_ids"])
                ).numpy(),
            }

            onnx_outputs = onnx_session.run(None, onnx_inputs)

            logger.info(
                f"ONNX inference successful, output shape: {onnx_outputs[1].shape}"
            )

            # Compare to PyTorch outputs
            with torch.no_grad():
                pytorch_outputs = model(
                    dummy_inputs["input_ids"], dummy_inputs["attention_mask"]
                )

            # Check outputs are close
            import numpy as np

            pytorch_pooler = pytorch_outputs.pooler_output.numpy()
            onnx_pooler = onnx_outputs[1]

            max_diff = np.max(np.abs(pytorch_pooler - onnx_pooler))
            logger.info(
                f"Maximum difference between PyTorch and ONNX outputs: {max_diff}"
            )

            if max_diff < 1e-4:
                logger.info("ONNX conversion successful!")
            else:
                logger.warning("ONNX outputs differ from PyTorch outputs")

        except Exception as e:
            logger.error(f"Error testing ONNX model: {str(e)}")

        logger.info(f"Model successfully converted and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        return False


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to ONNX format"
    )
    parser.add_argument(
        "--model", type=str, help="Name or path of the model to convert"
    )
    parser.add_argument("--output", type=str, help="Path to save the ONNX model")
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model for smaller size"
    )

    args = parser.parse_args()

    success = convert_to_onnx(
        model_name=args.model, output_path=args.output, quantize=args.quantize
    )

    if success:
        print("Model conversion successful")
        return 0
    else:
        print("Model conversion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
