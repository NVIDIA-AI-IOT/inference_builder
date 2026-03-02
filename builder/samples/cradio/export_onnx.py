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
import torch


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
    args = parser.parse_args()
    export_onnx(args.output_dir, args.resolution, args.model_path)
