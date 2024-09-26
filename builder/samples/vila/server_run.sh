#!/usr/bin/env bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################
set -e

# debug
# set -x

if [ -z "${TP_SIZE}" ]; then
    TP_SIZE=1
fi

if [ -z "${PARALLEL_SIZE}" ]; then
    PARALLEL_SIZE=4
fi

echo '
parameters:
{ key: "gpt_model_path"
value: {string_value: "'"${CHECKPOINTS_DIR}/${VILA_MODEL_NAME}/fp16/${TP_SIZE}-gpu"'" }}' >> /workspace/model_repo/vila1.5-13b/config.pbtxt

MAX_BATCH_SIZE=$(jq -r '.build_config.max_batch_size' ${CHECKPOINTS_DIR}/${VILA_MODEL_NAME}/fp16/${TP_SIZE}-gpu/config.json)

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

world_size=$(jq -r '.pretrained_config.mapping.world_size' ${CHECKPOINTS_DIR}/${VILA_MODEL_NAME}/fp16/${TP_SIZE}-gpu/config.json)
export WORLD_SIZE=$world_size

if [ $world_size -eq 1 ]; then
    command="tritonserver --model-repository=/workspace/model_repo --exit-timeout-secs=2 --http-header-forward-pattern '.*' --grpc-header-forward-pattern '.*' --log-verbose=$TRITION_VERBOSE --log-warning=true --log-error=true --log-info=true"
    python3 inference.py & eval "$command"
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
    python3 inference.py & eval "$command"
fi

