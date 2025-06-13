# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-devel-dev156" AS inference_base

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.7.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    transformers

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
   CUDA_MODULE_LOADING=LAZY CUDA_VER=12.8 make && \
   [ -f libnvds_infercustomparser_tao.so ] && echo "$PWD post processing build successful" || echo "$PWD post processing build falied" && \
   cp libnvds_infercustomparser_tao.so /opt/nvidia/deepstream/deepstream/lib && \
   cd ../../../ && \
   rm -rf ds_tao_apps_src

WORKDIR /workspace
ADD deepstream-app.tgz /workspace

## Set Environment variables
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
# Enable latency measurement
#ENV NVDS_ENABLE_LATENCY_MEASUREMENT=1
#ENV NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

ENTRYPOINT ["python3", "/workspace/__main__.py"]
#ENTRYPOINT ["/bin/bash"]
CMD []
