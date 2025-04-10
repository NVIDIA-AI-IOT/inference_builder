#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -t, --target     Select target pipeline (tao, changenet, gdino)"
    echo "  -m, --model      Select AI model type for validation:"
    echo "                   - For tao: rtdetr, cls, seg"
    echo "                   - For gdino: gdino, mgdino"
    echo "  -h, --help       Display this help message"
    echo
    echo "Examples:"
    echo "  $0                      # Build all pipeline targets"
    echo "  $0 -t tao -m rtdetr    # Build tao.tgz with RTDETR validation"
    echo "  $0 -t gdino -m gdino   # Build gdino.tgz with GDINO validation"
}

# Function to clean temporary directories
clean_tmp() {
    local target=$1
    local model=$2

    if [ -z "$target" ]; then
        echo "Cleaning all temporary directories..."
        rm -rf builder/samples/tao/validation/cls/.tmp
        rm -rf builder/samples/tao/validation/gdino/.tmp
        rm -rf builder/samples/tao/validation/mgdino/.tmp
        rm -rf builder/samples/tao/validation/rtdetr/.tmp
        rm -rf builder/samples/tao/validation/seg/.tmp
    else
        echo "Cleaning temporary directory for $model..."
        rm -rf "builder/samples/tao/validation/$model/.tmp"
    fi
}

# Build function with validation
build_target() {
    local target=$1
    local model=$2
    local ds_prefix="ds_${target}"
    
    echo "Building ${model} for ${target}..."
    
    # Base command that's common to all builds
    local cmd="python builder/main.py builder/samples/tao/${ds_prefix}.yaml \
        --server-type fastapi \
        -a builder/samples/tao/openapi.yaml \
        -o builder/samples/tao \
        -t"
    
    # Add validation if model is specified
    if [ ! -z "$model" ]; then
        cmd+=" -v builder/samples/tao/validation/${model}"
    fi
    
    # Add processors.py for gdino builds
    if [ "$target" = "gdino" ]; then
        cmd+=" -c builder/samples/tao/processors.py"
    fi
    
    eval "$cmd"
}

# Function to build all models for a target
build_all_models() {
    local target=$1
    case $target in
        "tao")
            for model in "rtdetr" "cls" "seg"; do
                build_target "$target" "$model"
            done
            ;;
        "gdino")
            for model in "gdino" "mgdino"; do
                build_target "$target" "$model"
            done
            ;;
        "changenet")
            build_target "$target" ""
            ;;
    esac
}

# Parse command line arguments
TARGET=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [ ! -z "$TARGET" ]; then
    case $TARGET in
        "tao")
            if [ ! -z "$MODEL" ] && [[ ! "$MODEL" =~ ^(rtdetr|cls|seg)$ ]]; then
                echo "Error: Invalid model type for tao. Valid options are: rtdetr, cls, seg"
                exit 1
            fi
            ;;
        "gdino")
            if [ ! -z "$MODEL" ] && [[ ! "$MODEL" =~ ^(gdino|mgdino)$ ]]; then
                echo "Error: Invalid model type for gdino. Valid options are: gdino, mgdino"
                exit 1
            fi
            ;;
        "changenet")
            if [ ! -z "$MODEL" ]; then
                echo "Note: changenet doesn't support model selection yet"
                exit 1
            fi
            ;;
        *)
            echo "Error: Invalid target. Valid options are: tao, changenet, gdino"
            exit 1
            ;;
    esac
fi

# Change to the inference-builder root directory (4 levels up from the script)
cd "$(dirname "$0")/../../../.." || exit 1

# Clean directories based on target and model
clean_tmp "$TARGET" "$MODEL"

# Build targets
if [ -z "$TARGET" ]; then
    # Build all targets with their respective validations
    build_all_models "tao"
    build_all_models "gdino"
    build_all_models "changenet"
else
    if [ -z "$MODEL" ]; then
        # Build all models for the specified target
        build_all_models "$TARGET"
    else
        # Build specific target with specific model
        build_target "$TARGET" "$MODEL"
    fi
fi
