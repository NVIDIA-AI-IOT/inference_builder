#!/usr/bin/env bash

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

set -e

# debug
# set -x

if [ -z "${TP_SIZE}" ]; then
    TP_SIZE=1
fi

if [ -z "${PARALLEL_SIZE}" ]; then
    PARALLEL_SIZE=4
fi

MAX_BATCH_SIZE=$(jq -r '.build_config.max_batch_size' /config/models/vila/vila1.5-13b/int4_awq/${TP_SIZE}-gpu/config.json)

echo '
instance_group [
    {
        count: '${PARALLEL_SIZE}'
        kind: KIND_MODEL
    }
]' >> /workspace/model_repo/vila/config.pbtxt


if [ -z "${VISION_BATCH_SIZE}" ]; then
    VISION_BATCH_SIZE=4
fi


if [ -z "${TMP_DIR}" ]; then
    TMP_DIR=/workspace/tmp
fi
echo "tmp folder: $TMP_DIR"

if [ -z "${TRITION_VERBOSE}" ]; then
    TRITION_VERBOSE=0
fi

world_size=$(jq -r '.pretrained_config.mapping.world_size' /config/models/vila/vila1.5-13b/int4_awq/${TP_SIZE}-gpu/config.json)
export WORLD_SIZE=$world_size

if [ $world_size -eq 1 ]; then
    command="tritonserver --model-repository=/workspace/model_repo --exit-timeout-secs=2 --http-header-forward-pattern '.*' --grpc-header-forward-pattern '.*' --log-verbose=$TRITION_VERBOSE --log-warning=true --log-error=true --log-info=true"
    python3 __main__.py & eval "$command"
else
    command='mpirun --allow-run-as-root'
    for i in $(seq 0 "$(($world_size-1))"); do
        if [ $i -eq 0 ]; then
            # First mpi rank should load all models
            loads='*'
        else
            # All others should only load vision and TRTLLM
            loads='tensorrt_llm'
        fi
        command="$command"" -n 1 tritonserver --model-repository=/workspace/model_repo --exit-timeout-secs=2 --model-control-mode=explicit --load-model=${loads} --backend-config=python,shm-region-prefix-name=prefix${i}_ --http-header-forward-pattern '.*' --grpc-header-forward-pattern '.*' --log-verbose=$TRITION_VERBOSE --log-warning=true --log-error=true --log-info=true :";
    done
    python3 __main__.py & eval "$command"
fi

