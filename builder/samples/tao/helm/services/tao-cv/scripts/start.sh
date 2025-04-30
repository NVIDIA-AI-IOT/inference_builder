#!/bin/bash

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
echo "Set NIM_MANIFEST_PATH to: $NIM_MANIFEST_PATH"

# Set NIM_CACHE_PATH
export NIM_CACHE_PATH="/opt/nim/.cache"
echo "Set NIM_CACHE_PATH to: $NIM_CACHE_PATH"

# Execute Docker entrypoint
/opt/nim/start_server.sh