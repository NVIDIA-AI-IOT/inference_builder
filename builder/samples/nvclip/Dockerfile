FROM gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:7.1.0-triton-blos-dev125 AS ds_stage

FROM "nvcr.io/nvidia/tritonserver:24.08-py3" AS nim_builder_base

COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/lib /opt/nvidia/deepstream/deepstream-7.1/lib
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/bin /opt/nvidia/deepstream/deepstream-7.1/bin
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.sh /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.txt /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/README* /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/*.pdf /opt/nvidia/deepstream/deepstream-7.1/
COPY --from=ds_stage /opt/nvidia/deepstream/deepstream-7.1/service-maker /opt/nvidia/deepstream/deepstream-7.1/service-maker

RUN /opt/nvidia/deepstream/deepstream-7.1/install.sh
RUN ln -sf deepstream-7.1 /opt/nvidia/deepstream/deepstream || true
RUN rm -rf /opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/libnvdsgst_udp.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstchromaprint.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpeg2enc.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpeg2dec.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstmpg123.so \
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstopenmpt.so

ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:/opt/tritonserver/lib/

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
    else \
        python3 -m pip install --upgrade pip; \
    fi \
# Remove vulnerabilities CVE-2022-29501
    && apt-get purge -y --auto-remove curl \
        libslurm37 \
        libpmi2-0 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y libyaml-cpp-dev libjson-glib-1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    libglew-dev \
    libssl-dev \
    libopencv-dev \
    freeglut3-dev \
    libjpeg-dev \
    libcurl4-gnutls-dev \
    libjson-glib-dev \
    libcairo2-dev \
    libpango1.0-dev  \
    libfontconfig1-dev \
    libfreetype6-dev \
    libgtk-3-dev \
    libpng-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    librabbitmq-dev
# Installing nimlib with pip assumes nimlib can be found at PIP_INDEX_URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv $NIM_DIR_PATH/.venv \
    && . $NIM_DIR_PATH/.venv/bin/activate \
    && pip install nimlib==0.1.30 \
    && pip install /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker-0.0.1-py3-none-linux_x86_64.whl \
    && pip install omegaconf==2.3.0
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
COPY data_model.py $NIM_DIR_PATH
COPY api.py $NIM_DIR_PATH
ADD common $NIM_DIR_PATH/common
ADD custom $NIM_DIR_PATH/custom
ADD model_repo $NIM_DIR_PATH/model_repo

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
    python3 /opt/nim/api.py & tritonserver --model-repository=/opt/nim/model_repo
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/opt/nim/start_server.sh"]
#ENTRYPOINT ["/bin/bash"]
CMD []q