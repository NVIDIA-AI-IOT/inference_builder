# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FROM "nvcr.io/nvidia/deepstream:8.0-triton-multiarch" AS inference_base

RUN --mount=type=cache,target=/root/.cache/pip pip install qwen-vl-utils[decord]==0.0.14 \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.11.0 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    vllm==0.11.0 \
    transformers==4.57.3

WORKDIR /workspace
ADD vllm.tgz /workspace

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/nvidia/deepstream/deepstream/lib
ENV NVSTREAMMUX_ADAPTIVE_BATCHING yes
ENV NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES,video
ENV TORCH_CUDA_ARCH_LIST "7.5 8.0 8.6 9.0 10.0"
ENV CUDA_CACHE_DISABLE 0

RUN touch /workspace/start_server.sh && \
    chmod a+rx /workspace/start_server.sh && \
    cat > /workspace/start_server.sh <<-EOF
	#!/usr/bin/env bash
    set -eu
	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    python3 /workspace/__main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/workspace/start_server.sh"]
CMD []
