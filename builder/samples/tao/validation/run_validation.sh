#!/bin/bash
set -e  # Exit on any error
# Validate NIM_MODEL_NAME
valid_models=("rtdetr" "cls" "seg" "gdino" "mgdino")
if [[ ! " ${valid_models[@]} " =~ " ${NIM_MODEL_NAME} " ]]; then
    echo "Error: NIM_MODEL_NAME must be one of: ${valid_models[*]}"
    exit 1
fi

# Change to the appropriate test directory
cd "/app/validation/${NIM_MODEL_NAME}/.tmp" || exit 1
echo $(pwd)
# Run the test
python test_runner.py
