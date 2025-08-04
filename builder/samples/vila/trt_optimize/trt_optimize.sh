#!/bin/bash

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
set -x

if [ -z "${LLM_BATCH_SIZE}" ]; then
    LLM_BATCH_SIZE=16
fi

if [ -z "${VISION_BATCH_SIZE}" ]; then
    VISION_BATCH_SIZE=4
fi

if [ -z "${VILA_40B_IMG_SIZE}" ]; then
    VILA_40B_IMG_SIZE=448
fi

WORKSPACE="/workspace"
VILA_DIR="$WORKSPACE/VILA"

if [ -z "${VILA_MODEL_NAME}" ]; then
    VILA_MODEL_NAME="vila1.5-13b"
fi

if [ -z "${TP_SIZE}" ]; then
    TP_SIZE=1
fi

if [ -z "${MODEL_OPT_DIR}" ]; then
    MODEL_OPT_DIR="${WORKSPACE}/checkpoints/optimized/${VILA_MODEL_NAME}"
fi
if [ -z "${MODEL_BASE_DIR}" ]; then
    MODEL_BASE_DIR="${WORKSPACE}/checkpoints/baseline/${VILA_MODEL_NAME}"
fi

# LLM_PRECISION: fp16, int4_awq
if [ -z "${LLM_PRECISION}" ]; then
    LLM_PRECISION=fp16
fi

echo "VILA_MODEL_NAME: ${VILA_MODEL_NAME}"
echo "LLM_BATCH_SIZE: ${LLM_BATCH_SIZE}"
echo "VISION_BATCH_SIZE: ${VISION_BATCH_SIZE}"
echo "MODEL_OPT_DIR: ${MODEL_OPT_DIR}"
echo "MODEL_BASE_DIR: ${MODEL_BASE_DIR}"
echo "LLM_PRECISION: ${LLM_PRECISION}"
echo "NEED_QUANTIZE: ${NEED_QUANTIZE}"
echo "SAFETENSOR_QUANT_DIR: ${SAFETENSOR_QUANT_DIR}"

trt_safetensor_dir=${MODEL_OPT_DIR}/trt_checkpoints/${LLM_PRECISION}/${TP_SIZE}-gpu

# quantize manually
if [ "X$NEED_QUANTIZE" = "X1" ]; then
    echo "quantizing model checkpoint $VILA_MODEL_NAME"
    if [ "X$LLM_PRECISION" = "fp16" ]; then
        LLM_PRECISION="int4_awq"
        echo "Updating LLM_PRECISION=int4_awq"
    fi
    if [ -z "${SAFETENSOR_QUANT_DIR}" ]; then
        SAFETENSOR_QUANT_DIR=${MODEL_BASE_DIR}/trt_checkpoints/${LLM_PRECISION}/${TP_SIZE}-gpu
    fi
    trt_safetensor_dir=${SAFETENSOR_QUANT_DIR}
    # convert engine
    cd "${WORKSPACE}"
    python quantize/quantize.py --model_dir ${MODEL_BASE_DIR} \
        --output_dir ${trt_safetensor_dir} \
        --dtype float16  --qformat ${LLM_PRECISION}   --calib_size 32 \
        || (echo "quantizing model failed."; exit 1)
    echo "[DONE]: quantizing model $VILA_MODEL_NAME"
elif [ -n "${SAFETENSOR_QUANT_DIR}" ]; then
    trt_safetensor_dir=${SAFETENSOR_QUANT_DIR}
    echo "Using SAFETENSOR_QUANT_DIR: ${SAFETENSOR_QUANT_DIR} directly..."
    echo "skipping checkpoint converting..."
else
    echo "converting model checkpoint $VILA_MODEL_NAME"
    # convert engine
    cd "${VILA_DIR}/demo_trt_llm"
    python3 convert_checkpoint.py \
        --model_dir ${MODEL_BASE_DIR} \
        --output_dir ${trt_safetensor_dir} \
        --dtype float16 \
        || (echo "converting model failed."; exit 1)
    echo "[DONE] converting model checkpoint $VILA_MODEL_NAME ..."
fi
echo "building trt-llm model $VILA_MODEL_NAME"

extra_options=""
if [ "${LLM_PRECISION}" = "fp16" ]; then
    extra_options+="--use_fused_mlp"
fi

echo "building trt-llm model $VILA_MODEL_NAME"
# build engine
trtllm-build \
    --checkpoint_dir ${trt_safetensor_dir} \
    --output_dir ${MODEL_OPT_DIR}/${LLM_PRECISION}/${TP_SIZE}-gpu \
    --gemm_plugin float16 \
    --max_batch_size ${LLM_BATCH_SIZE} \
    --max_input_len 4096 \
    --max_output_len 1024 \
    --max_multimodal_len $(( LLM_BATCH_SIZE * 4096 )) \
    ${extra_options} \
    || (echo "building trt-llm model failed."; exit 1)
echo "[DONE] building trt-llm engines $VILA_MODEL_NAME"

echo "building model  $VILA_MODEL_NAME vision encoder"

if [[ "$VILA_MODEL_NAME" == *"40b"* ]]; then
    python3 build_visual_engine.py \
        --model_path ${MODEL_BASE_DIR} \
        --model_type vila \
        --max_batch_size ${VISION_BATCH_SIZE} \
        --vila_path ${VILA_DIR} \
        --output_dir ${MODEL_OPT_DIR}/fp16/1-gpu/visual_engines \
        --export_onnx_only true \
        || (echo "converting $VILA_MODEL_NAME vision encoder to onnx failed."; exit 1)
    python3 build_visual_engine_from_onnx.py \
        --model_type vila \
        --height ${VILA_40B_IMG_SIZE} --width ${VILA_40B_IMG_SIZE} \
        --max_batch_size ${VISION_BATCH_SIZE} \
        --output_dir ${MODEL_OPT_DIR}/fp16/1-gpu/visual_engines \
        || (echo "converting $VILA_MODEL_NAME vision encoder to onnx failed."; exit 1)
else
    python3 build_visual_engine.py \
        --model_path ${MODEL_BASE_DIR} \
        --model_type vila \
        --max_batch_size ${VISION_BATCH_SIZE} \
        --vila_path ${VILA_DIR} \
        --output_dir ${MODEL_OPT_DIR}/fp16/1-gpu/visual_engines \
        || (echo "building $VILA_MODEL_NAME vision encoder failed."; exit 1)
fi
echo "[DONE] building model  $VILA_MODEL_NAME vision encoder"

echo "copying tokenizer from [$MODEL_BASE_DIR/llm] to optimized folder [${MODEL_OPT_DIR}]"
cp -ar ${MODEL_BASE_DIR}/llm/tokenizer* ${MODEL_OPT_DIR}/

echo "copying image processor from [$MODEL_BASE_DIR/vision_tower] to optimized folder [${MODEL_OPT_DIR}]"
cp -ar ${MODEL_BASE_DIR}/vision_tower/preprocessor_config.json ${MODEL_OPT_DIR}/

echo "deleting ${MODEL_OPT_DIR}/trt_checkpoints"
rm -rf ${MODEL_OPT_DIR}/trt_checkpoints

echo "[DONE] trt-optimization"
echo "Please upload folder [checkpoints/optimized] manually"
