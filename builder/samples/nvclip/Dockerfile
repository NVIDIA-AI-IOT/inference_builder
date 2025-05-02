FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-blos-dev271" AS nim_builder_base

ENV NIM_DIR_PATH="/opt/nim" \
    PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \python3-venv \
    && if ! command -v pip &> /dev/null; then \
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
        && python3 get-pip.py \
        && rm get-pip.py; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nimlib[runtime]==0.8.4 \
    && pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-py3-none-linux_x86_64.whl

# FROM "nvcr.io/nvidia/tritonserver:24.08-py3" AS nim_final

LABEL com.nvidia.nim.base_image="nvcr.io/nvidia/tritonserver:24.08-py3"
LABEL com.nvidia.nim.name=nvclip
LABEL com.nvidia.nim.type=triton
LABEL com.nvidia.nim.version=1.0.0
LABEL com.nvidia.nim.nspect="NSPECT-OJDQ-03HA"
LABEL com.nvidia.nim.inference_protocol=http

ENV NIM_CACHE_PATH="/opt/nim/.cache" \
    NIM_NAME=nvclip \
    PYTHONUNBUFFERED=1 \
    NIM_DIR_PATH="/opt/nim" \
    NGC_API_KEY=
ENV PATH="$NIM_DIR_PATH/.venv/bin:$PATH"
ENV BACKEND_TYPE="triton"
ENV BASE_IMAGE="nvcr.io/nvidia/tritonserver:24.08-py3"
ENV NIMTOOLS_VERSION=0.2.1rc1
ENV BACKEND_TYPE="triton"
ENV NIM_NSPECT_ID="NSPECT-OJDQ-03HA"
ENV NIM_MODEL_NAME="nvidia/nvclip-vit-h-14"
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV NIM_TRITON_LOG_VERBOSE=1
ENV TRANSFORMERS_CACHE=/tmp
ENV NIM_HTTP_API_PORT=8000


#COPY --from=nim_builder_base $NIM_DIR_PATH $NIM_DIR_PATH
COPY ./dependencies.yaml /etc/nim/config/dependencies.yaml


RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler

WORKDIR $NIM_DIR_PATH

# Copy data models.
ADD nvclip.tgz $NIM_DIR_PATH

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown 1000.1000 $NIM_DIR_PATH

USER nvs:1000

#  create entrypoint script at location indicated in NIM Playbook
RUN touch $NIM_DIR_PATH/start_server.sh && \
    chmod a+rx $NIM_DIR_PATH/start_server.sh && \
    cat > $NIM_DIR_PATH/start_server.sh <<-EOF
	#!/usr/bin/env bash
	set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    python3 /opt/nim/__main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/opt/nim/start_server.sh"]
#ENTRYPOINT ["/bin/bash"]
CMD []