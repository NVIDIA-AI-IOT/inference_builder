#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -xe

python3 export_to_onnx.py

mkdir -p /workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_vision
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_vision.onnx --saveEngine=/workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_vision/model.plan --optShapes=IMAGE:${MAX_BATCH_SIZE}x3x${INPUT_HEIGHT}x${INPUT_WIDTH} \
 --minShapes=IMAGE:1x3x${INPUT_HEIGHT}x${INPUT_WIDTH} --maxShapes=IMAGE:${MAX_BATCH_SIZE}x3x${INPUT_HEIGHT}x${INPUT_WIDTH} --fp16

mkdir -p /workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_text
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/checkpoints/baseline/nvclip_clipa_vit_h14_700M_text.onnx --saveEngine=/workspace/checkpoints/optimized/nvclip_clipa_vit_h14_700M_text/model.plan --optShapes=TEXT:${MAX_BATCH_SIZE}x${TEXT_LENGTH} \
 --minShapes=TEXT:1x${TEXT_LENGTH} --maxShapes=TEXT:${MAX_BATCH_SIZE}x${TEXT_LENGTH} --fp16


