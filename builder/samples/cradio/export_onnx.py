#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Export C-RADIOv3-H model from HuggingFace to ONNX format.

Usage:
    python export_onnx.py --output-dir /models/cradio_v3_h --resolution 256

After export, build a TensorRT engine:
    polygraphy convert /models/cradio_v3_h/model.onnx \
        --fp16 \
        -o /models/cradio_v3_h/model.plan
"""

import argparse
import os
import sys
import torch


def validate_directory_path(dir_path):
    """
    Validate directory path to prevent OS access violations.
    Args:
        dir_path: Directory path to validate
    Returns:
        str: Validated and sanitized directory path
    Raises:
        ValueError: If path is invalid or potentially dangerous
    """
    if not dir_path:
        raise ValueError("Directory path cannot be empty")

    try:
        abs_path = os.path.abspath(dir_path)
        resolved_path = os.path.realpath(abs_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid directory path: {e}")

    if ".." in os.path.normpath(dir_path):
        raise ValueError("Path traversal detected in directory path")

    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in dir_path for char in invalid_chars):
        raise ValueError("Invalid characters in directory path")

    system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot', '/usr/bin', '/usr/sbin']
    for sys_dir in system_dirs:
        if resolved_path.startswith(sys_dir):
            raise ValueError(f"Access to system directory {sys_dir} is not allowed")

    return resolved_path


def export_onnx(output_dir: str, resolution: int = 256, model_path: str = None):
    from transformers import AutoModel

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    source = model_path or "nvidia/C-RADIOv3-H"
    print(f"Loading C-RADIOv3-H model from {source}...")
    model = AutoModel.from_pretrained(
        source, trust_remote_code=True
    )
    model.eval().cuda()

    dummy_input = torch.randn(1, 3, resolution, resolution, device="cuda")

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["summary", "spatial_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "summary": {0: "batch_size"},
            "spatial_features": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved to {onnx_path}")
    print(
        f"\nNext step - build TensorRT engine:\n"
        f"  polygraphy convert {onnx_path} --fp16 -o {os.path.join(output_dir, 'model.plan')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export C-RADIOv3-H to ONNX")
    parser.add_argument(
        "--output-dir",
        default="/models/cradio_v3_h",
        help="Directory for output ONNX and TensorRT engine files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Input image resolution (default: 256)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local path to the model directory (skips HF download)",
    )
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Error: Argument parsing failed: {str(e)}")
        sys.exit(1)

    # Comprehensive security validation
    validation_errors = []

    # Validate and sanitize output directory
    try:
        args.output_dir = validate_directory_path(args.output_dir)
    except ValueError as e:
        validation_errors.append(f"Invalid output directory: {e}")

    # Validate and sanitize model path
    if args.model_path is not None:
        try:
            args.model_path = validate_directory_path(args.model_path)
        except ValueError as e:
            validation_errors.append(f"Invalid model path: {e}")

    # Exit if any validation errors
    if validation_errors:
        print("Security validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)

    export_onnx(args.output_dir, args.resolution, args.model_path)
