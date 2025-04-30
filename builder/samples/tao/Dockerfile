# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-blos-dev271" AS nim_builder_base

ENV NIM_DIR_PATH="/opt/nim" \
    PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3-distutils \
    && if ! command -v pip &> /dev/null; then \
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
        && python3 get-pip.py \
        && rm get-pip.py; \
    else \
        python3 -m pip install --upgrade pip; \
    fi

    #TODO: remove curl?
    #&& apt-get purge -y --auto-remove curl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nimlib[runtime]==0.8.4

LABEL com.nvidia.nim.base_image="gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-blos-dev271"
LABEL com.nvidia.nim.name={{MODEL_NAME}}
LABEL com.nvidia.nim.type=triton
LABEL com.nvidia.nim.version=0.0.1
LABEL com.nvidia.nim.nspect=NSPECT-Z39R-IVVG
LABEL com.nvidia.nim.inference_protocol=http

ENV NIM_CACHE_PATH="/opt/nim/.cache" \
    NIM_NAME=nv-tao-inference \
    PYTHONUNBUFFERED=1 \
    NGC_API_KEY=

ENV BACKEND_TYPE=triton
ENV BASE_IMAGE="nvcr.io/nvidia/tritonserver-pb24h1:24.03.04-py3"
ENV NIMTOOLS_VERSION=1.1.1
ENV BACKEND_TYPE="triton"
ENV NIM_NSPECT_ID=NSPECT-Z39R-IVVG
#ENV NIM_MODEL_NAME={{MODEL_NAME}}
ENV TRANSFORMERS_CACHE=/tmp

# COPY ./model_manifest.yaml /opt/nim/etc/default/model_manifest.yaml
COPY ./dependencies.yaml /etc/nim/config/dependencies.yaml

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler

    # Build TRT customized plugins
ARG TRT_VERSION_MAJOR=10
ARG TRT_VERSION_MINOR=3
ARG TRT_VERSION_PATCH=0
ARG TRT_VERSION_BUILD=26

ARG TRT_VERSION_MAJOR_MINOR=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR
ARG TRT_VERSION_MAJOR_MINOR_PATCH=$TRT_VERSION_MAJOR.$TRT_VERSION_MINOR.$TRT_VERSION_PATCH
ARG TRT_VERSION_FULL=$TRT_VERSION_MAJOR_MINOR_PATCH.$TRT_VERSION_BUILD

ARG CUDA_VERSION_MAJOR=12
ARG CUDA_VERSION_MINOR=6
ARG CUDA_VERSION_PATCH=37
ARG CUDA_VERSION_BUILD=001
ARG CUDA_VERSION_MAJOR_MINOR=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
ARG CUDA_VERSION_FULL=$CUDA_VERSION_MAJOR_MINOR.$CUDA_VERSION_PATCH.$CUDA_VERSION_BUILD
ARG CUDNN_VERSION=9.3.0.75

ENV TRT_VERSION=$TRT_VERSION_FULL+cuda$CUDA_VERSION_FULL

WORKDIR /tmp
ENV TRT_TAG="release/$TRT_VERSION_MAJOR_MINOR"
RUN mkdir trt_oss_src && \
   cd trt_oss_src && \
   echo "$PWD Building TRT OSS..." && \
   git clone -b $TRT_TAG https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p build && cd build && \
   cmake .. \
    -DGPU_ARCHS="80;86;90" \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    -DTRT_OUT_DIR=/tmp/out \
    -DCUDA_VERSION=$CUDA_VERSION_MAJOR_MINOR \
    -DCUDNN_VERSION=$CUDNN_VERSION \
    && for i in {1..50}; do make -j 12 && break; done
RUN   cp /tmp/out/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp /tmp/out/libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
   cp /tmp/out/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.$TRT_VERSION_MAJOR_MINOR_PATCH && \
   cp /tmp/out/trtexec /usr/local/bin/ && \
   cd ../../../ && \
   rm -rf trt_oss_src

# Build DS TAO app post processors
# need to be after DS SDK installation
ARG GITLAB_TOKEN
ENV DS_TAO_APPS_REPO=https://oauth2:${GITLAB_TOKEN}@gitlab-master.nvidia.com/DeepstreamSDK/deepstream_tao_apps.git
ENV DS_TAO_APPS_TAG="main"
ARG CACHE_BUSTER=unique_value_to_bypass_docker_cache
RUN mkdir ds_tao_apps_src && \
   cd ds_tao_apps_src && \
   echo "$PWD Pulling DS TAO apps repo..." && \
   git clone -b $DS_TAO_APPS_TAG $DS_TAO_APPS_REPO ds_tao_apps && \
   cd ds_tao_apps/post_processor && \
   echo "$PWD Building DS TAO apps post processing..." && \
   CUDA_MODULE_LOADING=LAZY CUDA_VER=$CUDA_VERSION_MAJOR_MINOR make && \
   [ -f libnvds_infercustomparser_tao.so ] && echo "$PWD post processing build successful" || echo "$PWD post processing build falied" && \
   cp libnvds_infercustomparser_tao.so /opt/nvidia/deepstream/deepstream/lib && \
   cd ../../../ && \
   rm -rf ds_tao_apps_src

FROM nim_builder_base AS nim_final

WORKDIR $NIM_DIR_PATH

# Patch JPEG Decoder
RUN if [ -e "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so" ]; then \
        mv /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so_bkp; \
    else \
        echo "File 'libgstnvcodec.so' does not exist"; \
    fi
RUN rm -rf /root/.cache/gstreamer-1.0/
RUN mkdir /tmp/assets && chmod -R 777 /tmp/assets

ADD tao.tgz $NIM_DIR_PATH

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown -R 1000.1000 $NIM_DIR_PATH && chmod o+w $NIM_DIR_PATH

USER nvs:1000

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
ENV NIM_DISABLE_TRITON_STARTUP=1

#  create entrypoint script at location indicated in NIM Playbook
RUN touch $NIM_DIR_PATH/start_server.sh && \
    chmod a+rx $NIM_DIR_PATH/start_server.sh && \
    cat > $NIM_DIR_PATH/start_server.sh <<-EOF
	#!/usr/bin/env bash
	set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    # Check if the symlink exists
    SYMLINK="\${NIM_CACHE_PATH}/model-repo/tao"
    if [ -L "\$SYMLINK" ]; then
        # If it exists, delete the symlink
        rm "\$SYMLINK"
        echo "Symlink deleted: \$SYMLINK"
    fi
    ln -s \${NIM_CACHE_PATH}/model-repo/\${NIM_MODEL_NAME} "\$SYMLINK"
    python3 __main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/opt/nim/start_server.sh"]
CMD []
