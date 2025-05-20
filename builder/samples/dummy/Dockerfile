# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-blos-dev271" AS inference_base

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends curl ca-certificates python3-distutils libyaml-cpp0.7 pkg-config cmake \
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
    pip install \
    python-multipart==0.0.9 \
    aiofiles==23.2.1 \
    aiohttp==3.8.5 \
    annotated-types==0.7.0 \
    python-magic==0.4.27 \
    torch==2.4.0 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    fastapi==0.115.12 \
    uvicorn==0.29.0

# Patch JPEG Decoder
RUN if [ -e "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so" ]; then \
        mv /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnvcodec.so_bkp; \
    else \
        echo "File 'libgstnvcodec.so' does not exist"; \
    fi
RUN rm -rf /root/.cache/gstreamer-1.0/
RUN mkdir /tmp/assets && chmod -R 777 /tmp/assets

WORKDIR /workspace
ADD dummy.tgz /workspace

RUN groupadd --gid 1000 --non-unique nvs && \
    useradd --create-home --shell /usr/sbin/nologin --uid 1000 --non-unique --gid 1000 nvs && \
    chown -R 1000.1000 /workspace && chmod o+w /workspace

USER nvs:1000

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes

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
