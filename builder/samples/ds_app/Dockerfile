# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-devel-dev122" AS inference_base

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.7.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0

WORKDIR /workspace
ADD deepstream-app.tgz /workspace

## Set Environment variables
ENV LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes
# Enable latency measurement
#ENV NVDS_ENABLE_LATENCY_MEASUREMENT=1
#ENV NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1

ENTRYPOINT ["python3", "/workspace/__main__.py"]
#ENTRYPOINT ["/bin/bash"]
CMD []
