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


# Test script wrapper for Docker container builds
# This script provides easy-to-use commands for testing Docker builds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_docker_builds.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    full-test      Run full test suite with all configurations from test_configs.json
    custom-test    Run test with custom configuration file
    help           Show this help message

Options:
    --no-cleanup           Don't cleanup Docker images after testing
    --output FILE          Save test report to specified file
    --config FILE          Use custom configuration file (required for custom-test)
    --log-dir DIR          Directory to save container logs (default: logs)

Examples:
    $0 full-test
    $0 custom-test --config my_config.json --output results.json
    $0 full-test --no-cleanup --output full_results.json --log-dir test_logs

EOF
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker is not running or you don't have permissions"
        exit 1
    fi

    # Check if Python script exists
    if [ ! -f "$TEST_SCRIPT" ]; then
        print_error "Test script not found: $TEST_SCRIPT"
        exit 1
    fi

    # Check if Python script is executable
    if [ ! -x "$TEST_SCRIPT" ]; then
        print_warning "Making test script executable..."
        chmod +x "$TEST_SCRIPT"
    fi

    print_success "Prerequisites check passed"
}

# Function to run full test
run_full_test() {
    print_info "Running full test suite with all configurations..."

    local args=("--dockerfile" "Dockerfile" "--base-dir" "." "--config-file" "test_configs.json" "--log-dir" "$LOG_DIR")

    if [ "$NO_CLEANUP" = "true" ]; then
        args+=("--no-cleanup")
    fi

    if [ -n "$OUTPUT_FILE" ]; then
        args+=("--output" "$OUTPUT_FILE")
    fi

    python3 "$TEST_SCRIPT" "${args[@]}"
}

# Function to run custom test
run_custom_test() {
    if [ -z "$CONFIG_FILE" ]; then
        print_error "Custom test requires --config option"
        exit 1
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi

    print_info "Running custom test with configuration: $CONFIG_FILE"

    local args=("--dockerfile" "Dockerfile" "--base-dir" "." "--config-file" "$CONFIG_FILE" "--log-dir" "$LOG_DIR")

    if [ "$NO_CLEANUP" = "true" ]; then
        args+=("--no-cleanup")
    fi

    if [ -n "$OUTPUT_FILE" ]; then
        args+=("--output" "$OUTPUT_FILE")
    fi

    python3 "$TEST_SCRIPT" "${args[@]}"
}

# Parse command line arguments
COMMAND=""
NO_CLEANUP="false"
OUTPUT_FILE=""
CONFIG_FILE=""
LOG_DIR="logs"

while [[ $# -gt 0 ]]; do
    case $1 in
        full-test|custom-test|help)
            COMMAND="$1"
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP="true"
            shift
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

# Handle help command
if [ "$COMMAND" = "help" ]; then
    show_usage
    exit 0
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Check prerequisites
check_prerequisites