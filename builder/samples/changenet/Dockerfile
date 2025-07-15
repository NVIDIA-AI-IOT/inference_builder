# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-25.07.1" AS builder_base

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    requests==2.32.3 \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.11.0 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    torch==2.4.0 \
    torchvision==0.19.0 \
    Pillow==11.1.0 \
    open_clip_torch==2.30.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    transformers==4.48.0 \
    fastapi==0.115.12 \
    uvicorn==0.29.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install inferencemodeltoolkit==1.0.11 --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

# CVCUDA for pre-processing
RUN wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-lib-0.15.0-cuda12-x86_64-linux.deb \
    https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-dev-0.15.0-cuda12-x86_64-linux.deb \
    https://github.com/CVCUDA/CV-CUDA/releases/download/v0.15.0-beta/cvcuda-python3.12-0.15.0-cuda11-x86_64-linux.deb \
    && dpkg -i *.deb

ENV TRANSFORMERS_CACHE=/tmp

WORKDIR /workspace
# Copy data models.
ADD changenet.tgz /workspace

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown 1000.1000 /workspace

USER nvs:1000

#  create entrypoint script at location indicated in NIM Playbook
RUN touch /workspace/start_server.sh && \
    chmod a+rx /workspace/start_server.sh && \
    cat > /workspace/start_server.sh <<-EOF
	#!/usr/bin/env bash
	set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    python3 /workspace/__main__.py & tritonserver --model-repository /workspace/model_repo
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/workspace/start_server.sh"]
#ENTRYPOINT ["/bin/bash"]
CMD []