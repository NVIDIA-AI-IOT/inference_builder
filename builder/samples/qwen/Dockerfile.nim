# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-devel-dev182" AS ds_stage

FROM nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3 AS inference_base
# Replace the base image with local built trtllm(v0.20.0rc3) image for using tensorrtllm/pytorch backend
# FROM tensorrt_llm/release:latest AS inference_base

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

# Comment out the following lines if using local built trtllm image
RUN --mount=type=cache,target=/root/.cache/pip pip install qwen-vl-utils[decord]==0.0.8
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/huggingface/transformers accelerate

ENV NIM_DIR_PATH="/opt/nim" \
    PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nimlib[runtime]==0.8.4

LABEL com.nvidia.nim.base_image="nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3"
LABEL com.nvidia.nim.name="qwenvl"
LABEL com.nvidia.nim.type=triton
LABEL com.nvidia.nim.version=0.0.1
LABEL com.nvidia.nim.nspect=NSPECT-Z39R-IVVG
LABEL com.nvidia.nim.inference_protocol=http

ENV NIM_CACHE_PATH="/opt/nim/.cache" \
    NIM_NAME=nv-tao-inference \
    PYTHONUNBUFFERED=1 \
    NGC_API_KEY=

ENV BACKEND_TYPE=triton
ENV BASE_IMAGE="nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3"
ENV NIMTOOLS_VERSION=1.1.1
ENV BACKEND_TYPE="triton"
ENV NIM_NSPECT_ID=NSPECT-Z39R-IVVG
ENV NIM_HTTP_API_PORT=8003
#ENV NIM_MODEL_NAME={{MODEL_NAME}}
ENV HF_HOME=/tmp

# COPY ./model_manifest.yaml /opt/nim/etc/default/model_manifest.yaml
COPY ./dependencies.yaml /etc/nim/config/dependencies.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler

WORKDIR $NIM_DIR_PATH
ADD qwen.tgz $NIM_DIR_PATH

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:/opt/nvidia/deepstream/deepstream/lib:/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
ENV HF_HOME=/tmp

RUN touch /opt/nim/start_server.sh && \
    chmod a+rx /opt/nim/start_server.sh && \
    cat > /opt/nim/start_server.sh <<-EOF
	#!/usr/bin/env bash
    set -eu
	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    python3 /opt/nim/inference.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/opt/nim/start_server.sh"]
CMD []
