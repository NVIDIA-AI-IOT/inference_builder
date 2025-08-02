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

# Check if NIM_MODEL_NAME is set
if [ -z "${NIM_MODEL_NAME:-}" ]; then
    echo "Error: NIM_MODEL_NAME environment variable is not set"
    echo "Please set NIM_MODEL_NAME to one of: rtdetr, cls, seg, gdino, mgdino"
    exit 1
fi

# Validate NIM_MODEL_NAME
valid_models=("rtdetr" "cls" "seg" "gdino" "mgdino" "changenet")
if [[ ! " ${valid_models[@]} " =~ " ${NIM_MODEL_NAME} " ]]; then
    echo "Error: Invalid NIM_MODEL_NAME: $NIM_MODEL_NAME"
    echo "Valid values are: ${valid_models[*]}"
    exit 1
fi

# Set NIM_MANIFEST_PATH based on NIM_MODEL_NAME
export NIM_MANIFEST_PATH="/opt/configs/model_manifest.${NIM_MODEL_NAME}.yaml"
echo "NIM_MANIFEST_PATH: $NIM_MANIFEST_PATH"

# Set NIM_CACHE_PATH
echo "NIM_CACHE_PATH: $NIM_CACHE_PATH"

# Execute Docker entrypoint
/opt/nim/start_server.sh