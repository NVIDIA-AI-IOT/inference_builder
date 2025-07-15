# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-25.07.1" AS ds_stage

FROM "nvcr.io/nvidia/tritonserver:25.01-py3" AS builder_base

# Install dependencies of DeepStream SDK for h/w decoder support
RUN apt update && apt install -y --no-install-recommends gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    libyaml-cpp-dev libgstrtspserver-1.0-0 libjsoncpp25 libgles2 pkg-config

COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/lib /opt/nvidia/deepstream/deepstream-8.0/lib
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/bin /opt/nvidia/deepstream/deepstream-8.0/bin
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/*.sh /opt/nvidia/deepstream/deepstream-8.0/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/*.txt /opt/nvidia/deepstream/deepstream-8.0/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/README* /opt/nvidia/deepstream/deepstream-8.0/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/*.pdf /opt/nvidia/deepstream/deepstream-8.0/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-8.0/service-maker /opt/nvidia/deepstream/deepstream-8.0/service-maker

RUN /opt/nvidia/deepstream/deepstream-8.0/install.sh
RUN ln -sf deepstream-8.0 /opt/nvidia/deepstream/deepstream || true

RUN --mount=type=cache,target=/root/.cache/pip \
pip install \
requests==2.32.3 \
python-multipart==0.0.9 \
aiofiles==23.2.1 \
aiohttp==3.11.0 \
annotated-types==0.7.0 \
python-magic==0.4.27 \
tensorrt==10.8.0.43 \
torch==2.4.0 \
torchvision==0.19.0 \
Pillow==11.1.0 \
open_clip_torch==2.30.0 \
numpy==1.26.4 \
omegaconf==2.3.0 \
transformers==4.48.0 \
fastapi==0.115.12 \
uvicorn==0.29.0 \
polygraphy==0.49.24 

ENV TRANSFORMERS_CACHE=/tmp

WORKDIR /workspace
# Copy data models.
ADD nvclip.tgz /workspace

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
    python3 /workspace/__main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/workspace/start_server.sh"]
#ENTRYPOINT ["/bin/bash"]
CMD []