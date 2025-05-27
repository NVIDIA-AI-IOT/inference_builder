# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

WORKDIR /workspace
ADD ds_app.tgz /workspace

## Set Environment variables
ENV LD_LIBRARY_PATH /opt/tritonserver/lib:$LD_LIBRARY_PATH
ENV NVSTREAMMUX_ADAPTIVE_BATCHING=yes

# Create the entrypoint to set up  the model path
RUN touch /workspace/start_server.sh && \
    chmod a+rx /workspace/start_server.sh && \
    cat > /workspace/start_server.sh <<-EOF
	#!/usr/bin/env bash
	set -eu

	# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
	# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
    # Check if the symlink exists
    SYMLINK="/workspace/models/ds"
    if [ -L "\$SYMLINK" ]; then
        # If it exists, delete the symlink
        rm "\$SYMLINK"
        echo "Symlink deleted: \$SYMLINK"
    fi

    ln -s /workspace/models/\${MODEL_NAME} "\$SYMLINK"
    python3 __main__.py
#    while true; do sleep 5; done
EOF

ENTRYPOINT ["/workspace/start_server.sh"]
CMD []
