# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-25.07.1" AS ds_stage

FROM nvcr.io/nvidia/tensorrt-llm/release:0.20.0 AS inference_base

ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

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

RUN --mount=type=cache,target=/root/.cache/pip pip install qwen-vl-utils[decord]==0.0.8 \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.11.0 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    numpy==1.26.4 \
    omegaconf==2.3.0

# Comment out the following lines if using local built trtllm image
RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/huggingface/transformers accelerate

WORKDIR /workspace
ADD qwen.tgz /workspace

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/opt/nvidia/deepstream/deepstream/lib:/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
ENV HF_HOME=/tmp

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
