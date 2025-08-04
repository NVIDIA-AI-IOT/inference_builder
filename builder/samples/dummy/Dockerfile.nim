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

FROM "gitlab-master.nvidia.com:5005/deepstreamsdk/release_image/deepstream:8.0.0-triton-blos-dev271" AS inference_base

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
    pip install nimlib[runtime]==0.8.4

ENV NIM_HTTP_API_PORT=8000

COPY ./dependencies.yaml /etc/nim/config/dependencies.yaml


RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler

WORKDIR $NIM_DIR_PATH

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    nim_dependency_handler

WORKDIR $NIM_DIR_PATH

# Copy data models.
ADD dummy.tgz $NIM_DIR_PATH

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