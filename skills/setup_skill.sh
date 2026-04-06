#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

# =============================================================================
# NVIDIA Inference Builder - Agent Skill Setup Script
# =============================================================================
#
# This script sets up the inference-builder Agent Skill by copying all
# necessary files including schemas and samples to a destination directory.
#
# Usage:
#   ./setup_skill.sh [destination_directory]
#
# If no destination is provided, defaults to ~/.claude/skills/inference-builder/
# which is Claude's personal skills directory.
#
# The script will create the following structure:
#   <destination>/
#   ├── SKILL.md                    # Skill documentation
#   ├── schemas/                    # JSON schemas
#   └── samples/                    # Sample resources organized by category
#       ├── config/                 # Pipeline YAML configs
#       ├── dockerfile/             # Dockerfiles
#       ├── processor/              # Pre/postprocessor Python files
#       ├── runtime_config/         # nvdsinfer_config.yaml files
#       ├── runtime_preprocess/     # nvdspreprocess_config*.yaml files
#       └── openapi/                # OpenAPI specs
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Read version from VERSION file
VERSION_FILE="${PROJECT_ROOT}/VERSION"
if [ ! -f "$VERSION_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} VERSION file not found: $VERSION_FILE"
    exit 1
fi
VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')

# Source directories
SCHEMAS_DIR="${PROJECT_ROOT}/schemas"
SAMPLES_DIR="${PROJECT_ROOT}/builder/samples"
SKILL_MD="${SCRIPT_DIR}/inference-builder/SKILL.md"

# Counters for summary
COUNT_SCHEMAS=0
COUNT_CONFIGS=0
COUNT_DOCKERFILES=0
COUNT_PROCESSORS=0
COUNT_RUNTIME_CONFIGS=0
COUNT_RUNTIME_PREPROCESS=0
COUNT_OPENAPI=0

# Default destination (Claude's personal skills directory)
DEFAULT_DEST="$HOME/.claude/skills"

# Skill folder name
SKILL_NAME="inference-builder"

# Function to print usage
print_usage() {
    echo "Usage: $0 [destination_directory]"
    echo ""
    echo "Set up the inference-builder Agent Skill at the specified location."
    echo "The script will create an 'inference-builder' subdirectory inside the destination."
    echo ""
    echo "Arguments:"
    echo "  destination_directory   Parent directory for the skill"
    echo "                          (default: ~/.claude/skills/)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Installs to ~/.claude/skills/inference-builder/"
    echo "  $0 ~/my-skills                        # Installs to ~/my-skills/inference-builder/"
    echo "  $0 /tmp/test                          # Installs to /tmp/test/inference-builder/"
}

# Function to log messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${CYAN}▶ $1${NC}"
}

# Validate source directories exist
validate_sources() {
    local errors=0

    if [ ! -f "$SKILL_MD" ]; then
        log_error "SKILL.md not found: $SKILL_MD"
        errors=$((errors + 1))
    fi

    if [ ! -d "$SCHEMAS_DIR" ]; then
        log_error "Schemas directory not found: $SCHEMAS_DIR"
        errors=$((errors + 1))
    fi

    if [ ! -d "$SAMPLES_DIR" ]; then
        log_error "Samples directory not found: $SAMPLES_DIR"
        errors=$((errors + 1))
    fi

    if [ $errors -gt 0 ]; then
        log_error "Source validation failed. Ensure you're running from the inference-builder project."
        exit 1
    fi
}

# Copy base skill files
copy_base_files() {
    log_section "Copying base skill files..."

    # Copy SKILL.md and stamp version from VERSION file
    sed "s/\${VERSION}/$VERSION/g" "$SKILL_MD" > "$DEST_DIR/SKILL.md"
    log_info "Copied SKILL.md (version: $VERSION)"

    # Create config file with project root, venv activation, and CLI path
    cat > "$DEST_DIR/.skill_config" << EOF
# Inference Builder Skill Configuration
# Generated by setup_skill.sh on $(date -Iseconds)
INFERENCE_BUILDER_VERSION="$VERSION"
INFERENCE_BUILDER_ROOT="$PROJECT_ROOT"
INFERENCE_BUILDER_VENV="source $PROJECT_ROOT/.venv/bin/activate"
INFERENCE_BUILDER_CLI="python $PROJECT_ROOT/builder/main.py"
EOF
    log_info "Created .skill_config (version: $VERSION, project root: $PROJECT_ROOT)"
}

# Copy schemas
copy_schemas() {
    log_section "Copying schemas..."

    mkdir -p "$DEST_DIR/schemas"

    # Copy main schema files
    for file in config.schema.json index.json README.md; do
        if [ -f "$SCHEMAS_DIR/$file" ]; then
            cp "$SCHEMAS_DIR/$file" "$DEST_DIR/schemas/"
            COUNT_SCHEMAS=$((COUNT_SCHEMAS + 1))
        fi
    done

    # Copy backends directory
    if [ -d "$SCHEMAS_DIR/backends" ]; then
        cp -r "$SCHEMAS_DIR/backends" "$DEST_DIR/schemas/"
        COUNT_SCHEMAS=$((COUNT_SCHEMAS + $(find "$SCHEMAS_DIR/backends" -type f | wc -l)))
    fi

    log_info "Copied schemas/ ($COUNT_SCHEMAS files)"
}

# Copy samples organized by category
copy_samples() {
    log_section "Copying samples..."

    # Create category directories
    mkdir -p "$DEST_DIR/samples/config"
    mkdir -p "$DEST_DIR/samples/dockerfile"
    mkdir -p "$DEST_DIR/samples/processor"
    mkdir -p "$DEST_DIR/samples/runtime_config"
    mkdir -p "$DEST_DIR/samples/runtime_preprocess"
    mkdir -p "$DEST_DIR/samples/openapi"

    # Process each sample directory
    for sample_dir in "$SAMPLES_DIR"/*/; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")

        # Skip non-directory items
        if [ "$sample_name" = "*" ]; then
            continue
        fi

        # Find and copy files by category
        while IFS= read -r -d '' file; do
            filename=$(basename "$file")
            rel_path="${file#$SAMPLES_DIR/}"

            # Determine category based on file type/name
            if [ "$filename" = "nvdsinfer_config.yaml" ]; then
                # Runtime config (nvinfer)
                dest_subdir="$DEST_DIR/samples/runtime_config/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_RUNTIME_CONFIGS=$((COUNT_RUNTIME_CONFIGS + 1))
            elif [[ "$filename" == nvdspreprocess_config* ]]; then
                # Runtime preprocess config
                dest_subdir="$DEST_DIR/samples/runtime_preprocess/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_RUNTIME_PREPROCESS=$((COUNT_RUNTIME_PREPROCESS + 1))
            elif [[ "$filename" == openapi.yaml ]] || [[ "$filename" == openapi.yml ]] || \
                 [[ "$filename" == *_openapi.yaml ]] || [[ "$filename" == *_openapi.yml ]]; then
                # OpenAPI specs
                dest_subdir="$DEST_DIR/samples/openapi/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_OPENAPI=$((COUNT_OPENAPI + 1))
            elif [[ "$filename" == Dockerfile* ]]; then
                # Dockerfiles
                dest_subdir="$DEST_DIR/samples/dockerfile/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_DOCKERFILES=$((COUNT_DOCKERFILES + 1))
            elif [[ "$filename" == *processor*.py ]] || [[ "$filename" == *processors*.py ]]; then
                # Processor files
                dest_subdir="$DEST_DIR/samples/processor/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_PROCESSORS=$((COUNT_PROCESSORS + 1))
            elif [[ "$filename" == *.yaml ]] || [[ "$filename" == *.yml ]]; then
                # Other YAML files are pipeline configs
                dest_subdir="$DEST_DIR/samples/config/$(dirname "$rel_path")"
                mkdir -p "$dest_subdir"
                cp "$file" "$dest_subdir/"
                COUNT_CONFIGS=$((COUNT_CONFIGS + 1))
            fi
        done < <(find "$sample_dir" -type f \( -name "*.yaml" -o -name "*.yml" -o -name "Dockerfile*" -o -name "*processor*.py" \) -print0)
    done

    # Copy sample README if exists
    if [ -f "$SAMPLES_DIR/README.md" ]; then
        cp "$SAMPLES_DIR/README.md" "$DEST_DIR/samples/"
    fi

    log_info "Copied samples/config/ ($COUNT_CONFIGS files)"
    log_info "Copied samples/dockerfile/ ($COUNT_DOCKERFILES files)"
    log_info "Copied samples/processor/ ($COUNT_PROCESSORS files)"
    log_info "Copied samples/runtime_config/ ($COUNT_RUNTIME_CONFIGS files)"
    log_info "Copied samples/runtime_preprocess/ ($COUNT_RUNTIME_PREPROCESS files)"
    log_info "Copied samples/openapi/ ($COUNT_OPENAPI files)"
}

# Print summary
print_summary() {
    local total_samples=$((COUNT_CONFIGS + COUNT_DOCKERFILES + COUNT_PROCESSORS + COUNT_RUNTIME_CONFIGS + COUNT_RUNTIME_PREPROCESS + COUNT_OPENAPI))
    local total_files=$((COUNT_SCHEMAS + total_samples + 2))

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Inference Builder Agent Skill v${VERSION} - Setup Complete${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}Location:${NC} $DEST_DIR"
    echo ""
    echo -e "  ${CYAN}Summary:${NC}"
    echo "    • SKILL.md:              1 file"
    echo "    • .skill_config:         1 file (project path)"
    echo "    • schemas/:              $COUNT_SCHEMAS files"
    echo "    • samples/config/:       $COUNT_CONFIGS files"
    echo "    • samples/dockerfile/:   $COUNT_DOCKERFILES files"
    echo "    • samples/processor/:    $COUNT_PROCESSORS files"
    echo "    • samples/runtime_config/:     $COUNT_RUNTIME_CONFIGS files"
    echo "    • samples/runtime_preprocess/: $COUNT_RUNTIME_PREPROCESS files"
    echo "    • samples/openapi/:      $COUNT_OPENAPI files"
    echo "    ─────────────────────────────────"
    echo "    Total:                   ~$total_files files"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}Structure:${NC}"
    tree -L 2 "$DEST_DIR" 2>/dev/null || find "$DEST_DIR" -maxdepth 2 -type d | head -20
    echo ""
}

# Main execution
main() {
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        print_usage
        exit 0
    fi

    # Use provided destination or default to Claude's skills directory
    if [ $# -eq 0 ]; then
        SKILLS_ROOT="$DEFAULT_DEST"
        log_info "No destination specified, using Claude's default: $SKILLS_ROOT"
    else
        SKILLS_ROOT="$1"
    fi

    # Append skill name to create full destination path
    DEST_DIR="${SKILLS_ROOT}/${SKILL_NAME}"

    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  NVIDIA Inference Builder - Agent Skill Setup${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""

    log_info "Project root: $PROJECT_ROOT"
    log_info "Skills directory: $SKILLS_ROOT"
    log_info "Skill destination: $DEST_DIR"

    # Validate sources
    validate_sources

    # Create destination directory
    if [ -d "$DEST_DIR" ]; then
        log_warn "Destination directory exists. Files may be overwritten."
    fi
    mkdir -p "$DEST_DIR"

    # Copy all components
    copy_base_files
    copy_schemas
    copy_samples

    # Print summary
    print_summary

    log_info "Setup complete! The skill is ready at: $DEST_DIR"
}

# Run main
main "$@"
