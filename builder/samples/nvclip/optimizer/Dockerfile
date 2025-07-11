# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FROM nvcr.io/nvidia/tritonserver:25.01-py3

WORKDIR /workspace/

# Environment variables
ENV PIP_NO_CACHE_DIR=off
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONUNBUFFERED=1

# Copy files into the container
COPY ./configs ./configs
COPY ./CLIP.png .
COPY ./export_to_onnx.py .
COPY ./optimize.sh .

ENV MAX_BATCH_SIZE=64
ENV INPUT_HEIGHT=224
ENV INPUT_WIDTH=224
ENV TEXT_LENGTH=77

# Make the script executable
RUN chmod +x /workspace/optimize.sh
RUN apt update
RUN apt-get install -y git
RUN pip install open_clip_torch onnx transformers polygraphy

# Set entrypoint to execute the installation script and then run optimize.sh
ENTRYPOINT ["/bin/bash", "-c", "/workspace/optimize.sh"]