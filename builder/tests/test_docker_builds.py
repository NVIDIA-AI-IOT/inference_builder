#!/usr/bin/env python3

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


"""
Test script for Docker container builds with different arguments.
This script tests the Dockerfile in the tests directory with various configurations.

Security Features:
- Input validation for app names and script commands to prevent command injection
- File path validation to prevent directory traversal attacks
- Safe subprocess calls using argument lists instead of shell string interpolation
- Logging of all executed commands for audit trails
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path
import shutil
import socket
from typing import Dict, List, Optional, Tuple
import logging
import re
import urllib.request
import urllib.error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_app_name(app_name: str) -> bool:
    """Validate app name to prevent command injection."""
    # Allow only alphanumeric characters, underscores, and hyphens
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', app_name))


def validate_script_command(script_command: str) -> bool:
    """Validate script command using a flexible approach that allows legitimate arguments."""
    if not isinstance(script_command, str) or not script_command.strip():
        return False

    script_command = script_command.strip()

    # Split command into parts for analysis
    try:
        import shlex
        parts = shlex.split(script_command)
    except ValueError:
        # Invalid shell syntax
        return False

    if not parts:
        return False

    # Check the main command/script
    main_command = parts[0]

    # Allow specific known scripts
    allowed_script_names = [
        "setup_rtsp_server.sh",
        "./setup_rtsp_server.sh"
    ]

    allowed_shell_commands = ["bash", "sh"]

    # Validate main command
    if main_command not in allowed_script_names and main_command not in allowed_shell_commands:
        return False

    # If it's a shell command, check the script being executed
    if main_command in allowed_shell_commands and len(parts) > 1:
        script_name = parts[1]
        if script_name not in allowed_script_names:
            return False

    # Validate all arguments
    for arg in parts[1:]:
        if not validate_script_arg(arg):
            return False

    # Check for dangerous patterns in the full command
    dangerous_patterns = [
        r'[;&|`]',        # Shell metacharacters (but allow some like spaces)
        r'\$\(',          # Command substitution
        r'`',             # Backticks
        r'>>?',           # Redirections
        r'\|\|',          # OR operator
        r'&&',            # AND operator
        r'<',             # Input redirection
        r'\x00',          # Null bytes
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, script_command):
            return False

    return True


def validate_script_arg(arg: str) -> bool:
    """Validate individual script argument."""
    if not isinstance(arg, str):
        return False

    # Check for null bytes and control characters
    if '\x00' in arg or any(ord(c) < 32 for c in arg if c not in ['\t']):
        return False

    # Allow common script arguments
    dangerous_chars = ['`', '$', ';', '|', '>', '<', '(', ')']
    if any(char in arg for char in dangerous_chars):
        return False

    # Allow legitimate flags and file arguments
    if arg.startswith('-'):
        # Allow common flag patterns
        if not re.match(r'^-{1,2}[a-zA-Z0-9][a-zA-Z0-9_-]*$', arg):
            return False

    # Prevent excessively long arguments
    if len(arg) > 1024:
        return False

    return True


def validate_safe_path(path: str) -> bool:
    """Validate file path to prevent directory traversal and command injection."""
    if not path or not isinstance(path, str):
        return False

    # Check for double slashes
    if '//' in path:
        return False

    # Check for absolute paths that could access system directories
    if os.path.isabs(path):
        system_dirs = ['/etc', '/sys', '/proc', '/dev', '/boot', '/usr/bin', '/usr/sbin', '/root']
        for sys_dir in system_dirs:
            if path.startswith(sys_dir):
                return False

    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\']
    if any(char in path for char in invalid_chars):
        return False

    return True

def validate_config_file_path(config_file: str) -> bool:
    """Validate configuration file path for security."""
    if not validate_safe_path(config_file):
        return False

    # Ensure the filename ends with .json
    filename = os.path.basename(config_file)
    if not filename.lower().endswith('.json'):
        return False

    return True

def validate_dockerfile_path(dockerfile: str) -> bool:
    """Validate Dockerfile path for security."""
    if not validate_safe_path(dockerfile):
        return False

    # Ensure the filename is Dockerfile or has .dockerfile extension
    filename = os.path.basename(dockerfile)
    if not (filename.lower() == 'dockerfile' or filename.lower().endswith('.dockerfile')):
        return False

    return True

def validate_log_directory(log_dir: str) -> bool:
    """Validate log directory path for security."""
    if not validate_safe_path(log_dir):
        return False

    # Prevent access to system directories
    if log_dir.startswith('/') and not log_dir.startswith('./'):
        return False

    return True

def validate_gitlab_token(token: str) -> bool:
    """Validate GitLab token format."""
    if not token:
        return True  # Empty token is allowed

    # Basic validation for GitLab token format
    if not isinstance(token, str) or len(token) < 10:
        return False

    # Check for suspicious patterns
    suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
    if any(pattern in token.lower() for pattern in suspicious_patterns):
        return False

    return True


def validate_docker_arg(arg: str) -> bool:
    """Validate docker argument to prevent command injection while allowing legitimate flags."""
    if not isinstance(arg, str):
        return False

    # Check for null bytes and control characters
    if '\x00' in arg or any(ord(c) < 32 for c in arg if c not in ['\t', '\n', '\r']):
        return False

    # Check for dangerous characters and patterns (but allow legitimate usage)
    dangerous_patterns = [
        r'[;|`$()]',      # Shell metacharacters (removed & for now)
        r'\$\(',          # Command substitution
        r'`',             # Backticks
        r'>>?',           # Redirections
        r'\|\|',          # OR operator
        r'&&',            # AND operator for command chaining
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # Control characters
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, arg):
            return False

    # Check for dangerous & usage (but allow in URL query parameters)
    if '&' in arg:
        # Allow & in URL-like contexts (after ?)
        if '?' not in arg:
            # & without ? suggests command chaining, not URL parameter
            return False
        # Check for command chaining patterns even with ?
        if ' &' in arg or '& ' in arg or arg.endswith('&'):
            return False

    # Additional dangerous characters for docker contexts (be selective)
    # Note: Removed '?' to allow URL query parameters, '*' and '!' for legitimate use
    dangerous_chars = ['[', ']', '{', '}', '\\']
    if any(char in arg for char in dangerous_chars):
        return False

    # Check for wildcard patterns that could be dangerous in specific contexts
    if '*' in arg and any(pattern in arg for pattern in ['*.*', '*.sh', '*.py', '*/', '/*']):
        return False

    # Allow legitimate command-line flags but prevent injection attempts
    if arg.strip().startswith('-'):
        # Allow common legitimate patterns
        legitimate_flag_patterns = [
            r'^--[a-zA-Z0-9][a-zA-Z0-9_-]*$',           # --flag-name
            r'^--[a-zA-Z0-9][a-zA-Z0-9_-]*=.*$',        # --flag=value
            r'^-[a-zA-Z0-9]$',                          # -f
            r'^-[a-zA-Z0-9][a-zA-Z0-9]*$',              # -abc
        ]

        # Check if it matches any legitimate pattern
        is_legitimate = any(re.match(pattern, arg) for pattern in legitimate_flag_patterns)

        if not is_legitimate:
            return False

        # Additional checks for flag values (after =)
        if '=' in arg:
            flag_value = arg.split('=', 1)[1]
            # Check flag value for dangerous patterns
            if any(char in flag_value for char in ['`', '$', ';', '&', '|', '(', ')']):
                return False

    # Prevent excessively long arguments (potential DoS)
    if len(arg) > 8192:  # Reasonable limit for docker arguments
        return False

    # Check for potential escape sequences in non-flag arguments
    if not arg.startswith('-') and '\\' in arg:
        if any(seq in arg for seq in ['\\n', '\\r', '\\t', '\\x', '\\u']):
            return False

    return True


def validate_env_var_name(name: str) -> bool:
    """Validate environment variable name."""
    if not isinstance(name, str) or not name:
        return False

    # Environment variable names should be alphanumeric + underscore, starting with letter or underscore
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def validate_env_var_value(value: str) -> bool:
    """Validate environment variable value."""
    if not isinstance(value, str):
        return False

    # Check for null bytes
    if '\x00' in value:
        return False

    # Check for dangerous shell characters
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<']
    if any(char in value for char in dangerous_chars):
        return False

    return True


def validate_volume_path(path: str) -> bool:
    """Validate volume mount path to prevent injection attacks.

    Note: Allows legitimate relative paths with .. for test configs.
    """
    if not isinstance(path, str) or not path:
        return False

    # Check for null bytes and control characters
    if '\x00' in path or any(ord(c) < 32 for c in path if c not in ['\t']):
        return False

    # Check for dangerous path patterns (but allow relative paths)
    # Allow: ../tao/, ./models, ../../shared
    # Reject: /../../../etc/passwd (absolute path traversal attacks)

    # Check for double slashes (suspicious)
    if '//' in path:
        return False

    # Check for Windows-style backslashes (not needed in Docker contexts)
    if '\\' in path:
        return False

    # Check for absolute path traversal attacks (starting with / and going up)
    # This would be trying to escape from an absolute path to system directories
    if path.startswith('/') and '/../' in path:
        # Check if trying to access system directories
        system_dirs = ['/../etc', '/../root', '/../sys', '/../proc', '/../dev']
        if any(path.startswith(sys_dir) for sys_dir in system_dirs):
            return False

    # Check for dangerous shell characters and command injection attempts
    # Allow ~ for home directory expansion at the start of path
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', '*', '?', '!', '[', ']', '{', '}']
    if any(char in path for char in dangerous_chars):
        return False

    # Allow ~ only at the start of path (for home directory expansion)
    if '~' in path and not path.startswith('~'):
        return False

    # Check for spaces at beginning/end (could be injection attempts)
    if path.startswith(' ') or path.endswith(' '):
        return False

    # Check for argument injection (starting with dash)
    if path.startswith('-'):
        return False

    # Ensure path doesn't contain colon (except for Windows drive letters or container paths)
    colon_count = path.count(':')
    if colon_count > 1:  # Allow one colon for Windows drive letters or container paths
        return False
    if colon_count == 1:
        # Allow Windows drive letters (C:) or absolute container paths (/app:)
        if not (re.match(r'^[A-Za-z]:', path) or ':' in path[1:]):
            return False

    # Additional check: ensure reasonable path length to prevent buffer overflow attacks
    if len(path) > 4096:  # Most systems limit paths to 4096 characters
        return False

    return True


def validate_build_arg_name(name: str) -> bool:
    """Validate Docker build argument name to prevent injection."""
    if not isinstance(name, str) or not name:
        return False

    # Build arg names should be alphanumeric + underscore, no dashes at start
    if name.startswith('-'):
        return False

    # Allow standard environment variable naming convention
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def validate_build_arg_value(value: str) -> bool:
    """Validate Docker build argument value to prevent injection."""
    if not isinstance(value, str):
        return False

    # Check for null bytes
    if '\x00' in value:
        return False

    # Check for dangerous shell characters that could cause issues
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', '\n', '\r']
    if any(char in value for char in dangerous_chars):
        return False

    # Check for argument injection attempts
    if value.strip().startswith('-'):
        return False

    return True


def validate_image_name(image_name: str) -> bool:
    """Validate Docker image name to prevent injection."""
    if not isinstance(image_name, str) or not image_name:
        return False

    # Check for null bytes
    if '\x00' in image_name:
        return False

    # Check for dangerous characters
    dangerous_chars = ['`', '$', ';', '&', '|', '(', ')', '>', '<', ' ', '\n', '\r']
    if any(char in image_name for char in dangerous_chars):
        return False

    # Basic docker image name validation (simplified)
    # Allow alphanumeric, hyphens, underscores, slashes, colons, dots
    if not re.match(r'^[a-zA-Z0-9._/-]+(?::[a-zA-Z0-9._-]+)?$', image_name):
        return False

    return True


def validate_test_config(test_config: dict) -> Tuple[bool, str]:
    """Validate test configuration to prevent command injection."""
    if not isinstance(test_config, dict):
        return False, "Test config must be a dictionary"

    # Validate environment variables
    if "env" in test_config:
        if not isinstance(test_config["env"], dict):
            return False, "env must be a dictionary"

        for key, value in test_config["env"].items():
            if not validate_env_var_name(key):
                return False, f"Invalid environment variable name: {key}"
            if not validate_env_var_value(str(value)):
                return False, f"Invalid environment variable value for {key}: {value}"

    # Validate volume mounts
    if "volumes" in test_config:
        if not isinstance(test_config["volumes"], dict):
            return False, "volumes must be a dictionary"

        for host_path, container_path in test_config["volumes"].items():
            if not validate_volume_path(host_path):
                return False, f"Invalid host path in volume mount: {host_path}"
            if not validate_volume_path(container_path):
                return False, f"Invalid container path in volume mount: {container_path}"

    # Validate command arguments
    if "cmd" in test_config:
        if not isinstance(test_config["cmd"], list):
            return False, "cmd must be a list"

        for arg in test_config["cmd"]:
            if not validate_docker_arg(str(arg)):
                return False, f"Invalid command argument: {arg}"

    # Validate timeout
    if "timeout" in test_config:
        if not isinstance(test_config["timeout"], (int, float)) or test_config["timeout"] <= 0:
            return False, "timeout must be a positive number"
        if test_config["timeout"] > 3600:  # Max 1 hour
            return False, "timeout cannot exceed 3600 seconds"

    # Validate prerequisite script
    if "prerequisite_script" in test_config:
        script = test_config["prerequisite_script"]
        if script and not validate_script_command(script):
            return False, f"Invalid prerequisite script: {script}"

    # Validate auto_validation configuration
    if "auto_validation" in test_config:
        if not isinstance(test_config["auto_validation"], str):
            return False, "auto_validation must be a string path"
        if not validate_safe_path(test_config["auto_validation"]):
            return False, f"Invalid auto_validation path: {test_config['auto_validation']}"
    # Validate payloads_path
    if "payloads_path" in test_config:
        payloads_path = test_config["payloads_path"]
        if not isinstance(payloads_path, str):
            return False, "payloads_path must be a string"
        if not validate_safe_path(payloads_path):
            return False, f"Invalid payloads_path: {payloads_path}"

    return True, ""


class DockerBuildTester:
    def __init__(self, dockerfile_path: str, base_dir: str, log_dir: Optional[str] = None):
        self.dockerfile_path = Path(dockerfile_path)
        self.base_dir = Path(base_dir)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.test_results = []

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)

    def download_models(self, models_config: Dict, config_dir: Path) -> Tuple[bool, str]:
        """Download models from NGC based on the models configuration.

        Args:
            models_config: Dictionary with model configurations
            config_dir: Directory containing the test config (for resolving relative paths)

        Returns:
            Tuple of (success, message)
        """
        if not models_config:
            return True, "No models to download"

        for model_name, model_info in models_config.items():
            try:
                source = model_info.get("source", "NGC")
                if source != "NGC":
                    logger.warning(f"⚠️  Unsupported model source '{source}' for {model_name}, skipping")
                    continue

                target_dir = Path(model_info["target"]).expanduser()
                model_path = model_info["path"]
                version = model_info["version"]
                configs_path = model_info.get("configs", "")

                # Construct the full NGC path
                ngc_path = f"{model_path}:{version}"

                # Determine the downloaded folder name (NGC naming convention)
                # e.g., "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0"
                # becomes "grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0"
                model_base_name = model_path.split('/')[-1]  # e.g., "grounding_dino"
                downloaded_folder = f"{model_base_name}_v{version}"

                # Final destination
                final_model_dir = target_dir / model_name

                # Check if model already exists
                if final_model_dir.exists():
                    logger.info(f"✅ Model '{model_name}' already exists at {final_model_dir}, skipping download")
                    continue

                logger.info(f"📥 Downloading model '{model_name}' from NGC...")
                logger.info(f"   NGC path: {ngc_path}")
                logger.info(f"   Target: {final_model_dir}")

                # Create target directory
                target_dir.mkdir(parents=True, exist_ok=True)

                # Download from NGC
                download_cmd = ["ngc", "registry", "model", "download-version", ngc_path]
                logger.info(f"   Running: {' '.join(download_cmd)}")

                result = subprocess.run(
                    download_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for download
                )

                if result.returncode != 0:
                    error_msg = f"Failed to download model '{model_name}': {result.stderr}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                # Move the downloaded folder to the target location
                downloaded_path = Path(downloaded_folder)
                if not downloaded_path.exists():
                    error_msg = f"Downloaded folder not found: {downloaded_path}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                logger.info(f"   Moving {downloaded_path} to {final_model_dir}")
                shutil.move(str(downloaded_path), str(final_model_dir))

                # Set permissions
                try:
                    os.chmod(final_model_dir, 0o777)
                    logger.info(f"   Set permissions: chmod 777 {final_model_dir}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to set permissions on {final_model_dir}: {e}")

                # Copy config files if specified
                if configs_path:
                    # Resolve configs_path relative to test config directory
                    source_configs = (config_dir / configs_path).resolve()
                    if source_configs.exists():
                        logger.info(f"   Copying configs from {source_configs} to {final_model_dir}")
                        # Copy all files from source_configs to final_model_dir
                        for item in source_configs.iterdir():
                            dest = final_model_dir / item.name
                            if item.is_file():
                                shutil.copy2(str(item), str(dest))
                            elif item.is_dir():
                                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                        logger.info(f"   ✅ Copied config files")
                    else:
                        logger.warning(f"⚠️  Config path not found: {source_configs}")

                logger.info(f"✅ Successfully downloaded and setup model '{model_name}'")

            except KeyError as e:
                error_msg = f"Missing required field in model config for '{model_name}': {e}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
            except subprocess.TimeoutExpired:
                error_msg = f"Model download timed out for '{model_name}'"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
            except Exception as e:
                error_msg = f"Exception during model download for '{model_name}': {str(e)}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

        return True, "All models downloaded successfully"

    def get_service_host(self) -> str:
        """
        Determine the appropriate host to use for connecting to Docker
        containers. In CI environments (like GitLab CI with
        Docker-in-Docker), 127.0.0.1 won't work because each container has
        its own localhost. Use Docker gateway IP instead.
        """
        # Check if running in CI environment
        if os.environ.get('CI') or os.environ.get('GITLAB_CI'):
            logger.info(
                "🔍 CI environment detected, using Docker gateway IP "
                "for service connectivity"
            )
            # Try to get docker gateway IP from bridge network
            try:
                result = subprocess.run(
                    [
                        "docker", "network", "inspect", "bridge", "-f",
                        "{{range .IPAM.Config}}{{.Gateway}}{{end}}"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    gateway_ip = result.stdout.strip()
                    logger.info(f"📡 Using Docker gateway IP: {gateway_ip}")
                    return gateway_ip
            except Exception as e:
                logger.warning(f"⚠️  Failed to get Docker gateway IP: {e}")

            # Fallback to default Docker bridge gateway
            logger.info(
                "📡 Using default Docker bridge gateway: 172.17.0.1"
            )
            return "172.17.0.1"

        # Local development environment - use localhost
        logger.info("📡 Using localhost for service connectivity")
        return "127.0.0.1"

    def generate_inference_code(self, build_args: Dict[str, str], test_config: Dict = None) -> Tuple[bool, str]:
        """Generate inference code (codegen) without building or testing Docker images.

        This runs the same pre-build code generation step used by build_image(),
        including optional OPENAPI_SPEC staging, but skips Docker build.

        Supports flexible path configuration via build_args:
        - APP_YAML_PATH: Custom path to app.yaml (overrides {TEST_APP_NAME}/app.yaml)
        - OUTPUT_DIR: Custom output directory (overrides {TEST_APP_NAME})
        - PROCESSORS_PATH: Custom processors.py path (overrides auto-detection)

        Args:
            build_args: Build arguments for code generation
            test_config: Optional test configuration. If present and contains 'auto_validation',
                        the validation directory will be passed to code generation via -v flag.
        """
        try:
            # Validate all build arguments to prevent command injection
            for key, value in build_args.items():
                if not validate_build_arg_name(key):
                    error_msg = f"Invalid build arg name: {key}. Only alphanumeric characters and underscores allowed."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg
                if not validate_build_arg_value(str(value)):
                    error_msg = f"Invalid build arg value for {key}: {value}. Value contains dangerous characters."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

            # Determine parameters for code generation
            test_app_name = build_args.get("TEST_APP_NAME", "frame_sampling")
            if not validate_app_name(test_app_name):
                error_msg = (
                    f"Invalid app name: {test_app_name}. Only alphanumeric characters, underscores, and hyphens are allowed."
                )
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            server_type = build_args.get("SERVER_TYPE", "serverless")
            openapi_spec = build_args.get("OPENAPI_SPEC")

            # Support flexible paths via build_args
            app_yaml_path = build_args.get("APP_YAML_PATH", f"{test_app_name}/app.yaml")
            output_dir = build_args.get("OUTPUT_DIR", test_app_name)
            processors_path_arg = build_args.get("PROCESSORS_PATH", "")

            # Find main.py relative to this script (in builder/main.py)
            main_py_path = Path(__file__).parent.parent / "main.py"
            if not main_py_path.exists():
                error_msg = f"Cannot find main.py at {main_py_path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            pre_build_command = [
                "python", str(main_py_path), app_yaml_path,
                "-o", output_dir
            ]

            # Add processors.py if specified or auto-detect
            if processors_path_arg:
                processors_path = Path(processors_path_arg)
                if processors_path.exists():
                    pre_build_command.extend(["-c", processors_path_arg])
                else:
                    logger.warning(f"⚠️  Specified PROCESSORS_PATH not found: {processors_path_arg}")
            else:
                # Auto-detect in output directory
                processors_path = Path(f"{test_app_name}/processors.py")
                if processors_path.exists():
                    pre_build_command.extend(["-c", f"{test_app_name}/processors.py"])

            # Add validation directory if specified in test_config
            if test_config and "auto_validation" in test_config:
                validation_folder = test_config["auto_validation"]
                # Resolve path relative to config directory if it's relative
                config_dir = test_config.get("_config_dir")
                if config_dir:
                    validation_path = (Path(config_dir) / validation_folder).resolve()
                else:
                    validation_path = Path(validation_folder).resolve()

                if validation_path.exists():
                    pre_build_command.extend(["-v", str(validation_path), "--no-docker"])
                    logger.info(f"📁 Adding validation directory for build: {validation_path}")
                else:
                    logger.warning(f"⚠️  Auto validation folder not found: {validation_path}")

            pre_build_command.extend(["--server-type", server_type, "-t"])

            if openapi_spec:
                # Resolve provided spec relative to project root, copy into local app folder to avoid unsafe paths
                project_root = Path(__file__).resolve().parents[2]
                resolved_spec = None
                candidates = []
                spec_path = Path(openapi_spec)
                if spec_path.is_absolute():
                    candidates.append(spec_path)
                else:
                    candidates.append((project_root / openapi_spec).resolve())
                    candidates.append((project_root / "builder" / openapi_spec).resolve())

                for cand in candidates:
                    try:
                        if cand.exists() and str(cand).startswith(str(project_root)):
                            resolved_spec = cand
                            break
                    except Exception:
                        continue

                if not resolved_spec:
                    error_msg = f"Invalid OPENAPI_SPEC path: {openapi_spec}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                local_spec_path = Path(output_dir) / "openapi.yaml"

                # Only copy if source and destination are different
                if resolved_spec.resolve() != local_spec_path.resolve():
                    try:
                        shutil.copyfile(str(resolved_spec), str(local_spec_path))
                        logger.info(f"📄 Staged OPENAPI_SPEC: {resolved_spec} -> {local_spec_path}")
                    except Exception as e:
                        error_msg = f"Failed to stage OPENAPI_SPEC: {e}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg
                else:
                    logger.info(f"📄 OPENAPI_SPEC already in output directory: {local_spec_path}")

                pre_build_command[3:3] = ["-a", str(local_spec_path)]

            logger.info(f"🔧 Executing codegen command: {' '.join(pre_build_command)}")
            pre_build_result = subprocess.run(
                pre_build_command,
                capture_output=True,
                text=True,
                timeout=600
            )

            if pre_build_result.returncode != 0:
                error_msg = f"Code generation failed: {pre_build_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info("✅ Code generation completed successfully")
            if pre_build_result.stdout:
                logger.info(f"Codegen output: {pre_build_result.stdout}")
            return True, pre_build_result.stdout

        except subprocess.TimeoutExpired:
            error_msg = "Code generation timed out"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during code generation: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def build_image(self, build_args: Dict[str, str], image_name: str, dockerfile: Optional[str] = None, base_dir: Optional[str] = None, test_config: Dict = None) -> Tuple[bool, str]:
        """Build Docker image with given arguments.

        Supports flexible path configuration via build_args:
        - APP_YAML_PATH: Custom path to app.yaml (overrides {TEST_APP_NAME}/app.yaml)
        - OUTPUT_DIR: Custom output directory (overrides {TEST_APP_NAME})
        - PROCESSORS_PATH: Custom processors.py path (overrides auto-detection)

        Args:
            build_args: Build arguments for the Docker image
            image_name: Name for the Docker image
            dockerfile: Optional path to Dockerfile (overrides self.dockerfile_path)
            base_dir: Optional Docker build context directory (overrides self.base_dir)
            test_config: Optional test configuration. If present and contains 'auto_validation',
                        the validation directory will be passed to code generation via -v flag.
        """
        try:
            # Use provided paths or fall back to instance defaults
            dockerfile_path = Path(dockerfile) if dockerfile else self.dockerfile_path
            build_context = Path(base_dir) if base_dir else self.base_dir
            # Validate image name to prevent command injection
            if not validate_image_name(image_name):
                error_msg = f"Invalid image name: {image_name}. Image name contains invalid characters."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Validate all build arguments to prevent command injection
            for key, value in build_args.items():
                if not validate_build_arg_name(key):
                    error_msg = f"Invalid build arg name: {key}. Only alphanumeric characters and underscores allowed."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg
                if not validate_build_arg_value(str(value)):
                    error_msg = f"Invalid build arg value for {key}: {value}. Value contains dangerous characters."
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

            # Execute pre-build command using TEST_APP_NAME
            test_app_name = build_args.get("TEST_APP_NAME", "frame_sampling")

            # Validate app name to prevent command injection
            if not validate_app_name(test_app_name):
                error_msg = f"Invalid app name: {test_app_name}. Only alphanumeric characters, underscores, and hyphens are allowed."
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Determine server type from build args (fallback to 'serverless')
            server_type = build_args.get("SERVER_TYPE", "serverless")

            # Optional: OpenAPI spec path (to pass -a)
            openapi_spec = build_args.get("OPENAPI_SPEC")

            # Require OpenAPI spec for non-serverless builds
            if server_type != "serverless" and not openapi_spec:
                error_msg = (
                    "OPENAPI_SPEC is required in build_args when SERVER_TYPE is not 'serverless'"
                )
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Support flexible paths via build_args
            app_yaml_path = build_args.get("APP_YAML_PATH", f"{test_app_name}/app.yaml")
            output_dir = build_args.get("OUTPUT_DIR", test_app_name)
            processors_path_arg = build_args.get("PROCESSORS_PATH", "")

            # Find main.py relative to this script (in builder/main.py)
            main_py_path = Path(__file__).parent.parent / "main.py"
            if not main_py_path.exists():
                error_msg = f"Cannot find main.py at {main_py_path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Use safer subprocess call without shell=True
            pre_build_command = [
                "python", str(main_py_path), app_yaml_path,
                "-o", output_dir
            ]

            # Add processors.py if specified or auto-detect
            if processors_path_arg:
                processors_path = Path(processors_path_arg)
                if processors_path.exists():
                    pre_build_command.extend(["-c", processors_path_arg])
                else:
                    logger.warning(f"⚠️  Specified PROCESSORS_PATH not found: {processors_path_arg}")
            else:
                # Auto-detect in test_app_name directory
                processors_path = Path(f"{test_app_name}/processors.py")
                if processors_path.exists():
                    pre_build_command.extend(["-c", f"{test_app_name}/processors.py"])

            # Add validation directory if specified in test_config
            if test_config and "auto_validation" in test_config:
                validation_folder = test_config["auto_validation"]
                # Resolve path relative to config directory if it's relative
                config_dir = test_config.get("_config_dir")
                if config_dir:
                    validation_path = (Path(config_dir) / validation_folder).resolve()
                else:
                    validation_path = Path(validation_folder).resolve()

                if validation_path.exists():
                    pre_build_command.extend(["-v", str(validation_path), "--no-docker"])
                    logger.info(f"📁 Adding validation directory for build: {validation_path}")
                else:
                    logger.warning(f"⚠️  Auto validation folder not found: {validation_path}")

            pre_build_command.extend(["--server-type", server_type, "-t"])

            if openapi_spec:
                # Resolve provided spec relative to project root, copy into local app folder to avoid unsafe paths
                project_root = Path(__file__).resolve().parents[2]
                resolved_spec = None
                candidates = []
                spec_path = Path(openapi_spec)
                if spec_path.is_absolute():
                    candidates.append(spec_path)
                else:
                    candidates.append((project_root / openapi_spec).resolve())
                    candidates.append((project_root / "builder" / openapi_spec).resolve())

                for cand in candidates:
                    try:
                        if cand.exists() and str(cand).startswith(str(project_root)):
                            resolved_spec = cand
                            break
                    except Exception:
                        continue

                if not resolved_spec:
                    error_msg = f"Invalid OPENAPI_SPEC path: {openapi_spec}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg

                local_spec_path = Path(output_dir) / "openapi.yaml"

                # Only copy if source and destination are different
                if resolved_spec.resolve() != local_spec_path.resolve():
                    try:
                        shutil.copyfile(str(resolved_spec), str(local_spec_path))
                        logger.info(f"📄 Staged OPENAPI_SPEC: {resolved_spec} -> {local_spec_path}")
                    except Exception as e:
                        error_msg = f"Failed to stage OPENAPI_SPEC: {e}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg
                else:
                    logger.info(f"📄 OPENAPI_SPEC already in output directory: {local_spec_path}")

                pre_build_command[3:3] = ["-a", str(local_spec_path)]

            logger.info(f"🔧 Executing pre-build command: {' '.join(pre_build_command)}")
            pre_build_result = subprocess.run(
                pre_build_command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for pre-build
            )

            if pre_build_result.returncode != 0:
                error_msg = f"Pre-build command failed: {pre_build_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info("✅ Pre-build command completed successfully")
            if pre_build_result.stdout:
                logger.info(f"Pre-build output: {pre_build_result.stdout}")

            cmd = [
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", image_name
            ]

            # Add build arguments
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

            # Add context directory
            cmd.append(str(build_context))

            logger.info(f"Building image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"✅ Successfully built image: {image_name}")
                return True, result.stdout
            else:
                logger.error(f"❌ Failed to build image: {image_name}")
                logger.error(f"Error: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Build timed out for image: {image_name}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during build: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def run_prerequisite_script(self, script_command: str, test_id: int) -> Tuple[bool, str]:
        """Run a prerequisite script before testing the docker container."""
        if not script_command:
            return True, "No prerequisite script specified"

        # Validate script command to prevent command injection
        if not validate_script_command(script_command):
            error_msg = f"Invalid script command: {script_command}. Command contains potentially dangerous characters."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        log_file = self.log_dir / f"prerequisite_{test_id}.log"

        try:
            logger.info(f"🔧 Running prerequisite script: {script_command}")
            logger.info(f"📄 Prerequisite logs will be saved to: {log_file}")

            # Use safer command execution - split command into arguments when possible
            import shlex
            try:
                # Attempt to parse command safely first
                cmd_args = shlex.split(script_command)
                if len(cmd_args) == 1 and cmd_args[0].endswith('.sh'):
                    # Single script file - use direct execution without shell
                    result = subprocess.run(
                        cmd_args,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        shell=False  # Safer execution without shell
                    )
                else:
                    # Complex command - use shell with additional validation
                    # Double-check validation before shell execution
                    if not validate_script_command(script_command):
                        raise ValueError("Command failed security validation")

                    result = subprocess.run(
                        script_command,
                        shell=True,  # Only when necessary and after validation
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
            except ValueError as e:
                if "failed security validation" in str(e):
                    raise e
                # Fall back to shell execution for complex commands (with validation)
                result = subprocess.run(
                    script_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

            # Save prerequisite script logs
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            if result.returncode == 0:
                logger.info(f"✅ Prerequisite script completed successfully")
                if result.stdout:
                    logger.info(f"Prerequisite output: {result.stdout}")
                return True, result.stdout
            else:
                error_msg = f"Prerequisite script failed with return code {result.returncode}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"Prerequisite stderr: {result.stderr}")
                return False, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"Prerequisite script timed out after 300 seconds"
            logger.error(f"❌ {error_msg}")

            # Save timeout log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: 300 seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg
        except Exception as e:
            error_msg = f"Exception during prerequisite script execution: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Prerequisite Script Execution ===\n")
                f.write(f"Command: {script_command}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n=== END LOG ===\n")

            return False, error_msg

    def test_image(self, image_name: str, test_config: Dict, test_id: int) -> Tuple[bool, str, str]:
        """Test the built image by running it and capture logs."""
        log_file = self.log_dir / f"test_{test_id}_{image_name.replace(':', '_')}.log"

        # Validate image name to prevent command injection
        if not validate_image_name(image_name):
            error_msg = f"Invalid image name: {image_name}. Image name contains invalid characters."
            logger.error(f"❌ {error_msg}")
            return False, error_msg, ""

        # Validate test configuration to prevent command injection
        config_valid, config_error = validate_test_config(test_config)
        if not config_valid:
            error_msg = f"Invalid test configuration: {config_error}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, ""

        # Get timeout from test config, default to 10 seconds
        timeout = test_config.get("timeout", 10)

        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return False, f"Image {image_name} not found", ""

            # Run prerequisite script if specified
            prerequisite_script = test_config.get("prerequisite_script")
            if prerequisite_script:
                prerequisite_success, prerequisite_output = self.run_prerequisite_script(
                    prerequisite_script, test_id
                )
                if not prerequisite_success:
                    return False, f"Prerequisite script failed: {prerequisite_output}", ""

            # Detect server type to choose test strategy
            server_type = test_config.get("SERVER_TYPE", "serverless")

            # Run the container with test configuration
            # Use host network for serverless; use port mapping for non-serverless to avoid port conflicts
            cmd = ["docker", "run", "--gpus", "all"]

            # Add environment variables if specified
            if "env" in test_config:
                for key, value in test_config["env"].items():
                    # Double-check validation to prevent injection
                    if not validate_env_var_name(key) or not validate_env_var_value(str(value)):
                        error_msg = f"Invalid environment variable: {key}={value}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""
                    cmd.extend(["-e", f"{key}={value}"])

            # Add volume mounts if specified
            if "volumes" in test_config:
                logger.info(f"📁 Processing volumes: {test_config['volumes']}")
                # Get the test config directory for resolving relative paths
                config_dir = Path(test_config.get("_config_dir", ".")).resolve()

                for host_path, container_path in test_config["volumes"].items():
                    # Double-check validation to prevent injection
                    if not validate_volume_path(host_path) or not validate_volume_path(container_path):
                        error_msg = f"Invalid volume path: {host_path}:{container_path}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""

                    # Resolve host path to absolute path
                    # Priority: 1) Expand ~ for home directory, 2) Resolve relative to config dir, 3) Use as-is if absolute
                    expanded_path = os.path.expanduser(host_path)

                    if not os.path.isabs(expanded_path):
                        # Relative path - resolve relative to test config directory
                        abs_host_path = str((config_dir / expanded_path).resolve())
                        logger.info(f"📁 Resolved relative path: {host_path} -> {abs_host_path} (relative to config dir)")
                    else:
                        # Already absolute path
                        abs_host_path = expanded_path
                        logger.info(f"📁 Using absolute path: {host_path} -> {abs_host_path}")

                    # Verify the resolved path exists (optional warning, not error)
                    if not os.path.exists(abs_host_path):
                        logger.warning(f"⚠️  Volume path does not exist yet: {abs_host_path}")
                        logger.warning(f"⚠️  This is OK if the path will be created by model download or other setup")

                    volume_arg = f"{abs_host_path}:{container_path}"
                    cmd.extend(["-v", volume_arg])
                    logger.info(f"📁 Adding volume mount: {volume_arg}")
            else:
                logger.info("📁 No volumes specified in test config")

            # For FastAPI (or any non-serverless), run detached on host network (DinD friendly)
            if server_type != "serverless":
                cmd.insert(2, "--network=host")
                cmd.insert(2, "-d")  # run detached
            else:
                cmd.insert(2, "--network=host")

            cmd.append(image_name)

            # Add command arguments if specified (serverless only)
            if server_type == "serverless" and "cmd" in test_config:
                # Double-check validation for each command argument
                for arg in test_config["cmd"]:
                    if not validate_docker_arg(str(arg)):
                        error_msg = f"Invalid command argument: {arg}"
                        logger.error(f"❌ {error_msg}")
                        return False, error_msg, ""
                cmd.extend(test_config["cmd"])

            logger.info(f"Testing image: {image_name}")
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Timeout: {timeout} seconds")
            logger.info(f"Logs will be saved to: {log_file}")

            # Log the complete command for debugging
            logger.info("🔍 Complete docker run command:")
            logger.info(f"   {' '.join(cmd)}")

            # Check if error detection is enabled
            error_detection_config = test_config.get("error_detection", {})
            error_detection_enabled = error_detection_config.get("enabled", False)

            if server_type == "serverless":
                if error_detection_enabled:
                    # Capture output for error detection
                    logger.info("🔍 Error detection enabled - capturing output for analysis")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout  # Use configurable timeout
                    )
                    stdout_output = result.stdout
                    stderr_output = result.stderr
                else:
                    # Stream output to host stdout in real-time
                    logger.info("📺 Error detection disabled - streaming output to host stdout")
                    logger.info("=" * 80)
                    logger.info(f"CONTAINER OUTPUT FOR: {image_name}")
                    logger.info("=" * 80)

                    # Run container and stream output to host stdout in real-time
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                        text=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )

                    # Collect output while streaming to stdout
                    stdout_output = ""
                    try:
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                print(output, end='', flush=True)
                                stdout_output += output

                        # Wait for process to complete and get return code
                        return_code = process.poll()
                        stderr_output = ""

                    except KeyboardInterrupt:
                        process.terminate()
                        process.wait()
                        return_code = process.poll()
                        stderr_output = ""
                        logger.info("\n⚠️  Process interrupted by user")

                    # Create result object for consistency
                    class Result:
                        def __init__(self, returncode, stdout, stderr):
                            self.returncode = returncode
                            self.stdout = stdout
                            self.stderr = stderr

                    result = Result(return_code, stdout_output, stderr_output)

                    logger.info("=" * 80)
                    logger.info(f"END CONTAINER OUTPUT FOR: {image_name}")
                    logger.info("=" * 80)
            else:
                # FastAPI-like server flow: start container detached, poll readiness, run client, then stop and collect logs
                logger.info("🚀 Starting server container in detached mode")
                start_proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                if start_proc.returncode != 0:
                    error_msg = f"Failed to start server container: {start_proc.stderr}"
                    logger.error(f"❌ {error_msg}")
                    return False, error_msg, ""
                container_id = start_proc.stdout.strip()
                logger.info(f"🆔 Container ID: {container_id}")

                # Readiness probe
                ready = False
                ready_deadline = time.time() + max(1, timeout)
                service_host = self.get_service_host()
                health_url = f"http://{service_host}:8000/v1/health/ready"
                logger.info(f"🔎 Probing readiness: {health_url}")
                while time.time() < ready_deadline:
                    try:
                        with urllib.request.urlopen(health_url, timeout=2) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                            else:
                                logger.info(f"Health probe returned non-200 status: {resp.status}")
                    except urllib.error.HTTPError as e:
                        logger.warning(f"Health probe HTTPError: status={e.code}")
                    except urllib.error.URLError as e:
                        if isinstance(e.reason, socket.timeout):
                            logger.warning("Health probe request timed out")
                        else:
                            logger.info(f"Health probe URLError: {e.reason}, retrying...")
                    time.sleep(2)

                client_stdout = ""
                client_stderr = ""
                client_rc = 1

                if not ready:
                    logger.error("❌ Server did not become ready within timeout")
                    # Collect logs to see what went wrong during initialization
                    logger.info("📋 Collecting container logs for failed readiness...")
                    logs_proc = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
                    if logs_proc.stdout:
                        logger.info("🔍 Container initialization logs:")
                        for line in logs_proc.stdout.split('\n')[-20:]:  # Show last 20 lines
                            if line.strip():
                                logger.info(f"   {line}")
                    if logs_proc.stderr:
                        logger.error("🔍 Container error logs:")
                        for line in logs_proc.stderr.split('\n')[-10:]:  # Show last 10 error lines
                            if line.strip():
                                logger.error(f"   {line}")
                else:
                    # Check if auto_validation is specified - run directly on host
                    auto_validation_path = test_config.get("auto_validation")
                    if auto_validation_path:
                        logger.info("✅ Server is ready. Running validation script on host...")

                        try:
                            # Resolve validation folder path relative to config directory
                            config_dir = Path(test_config.get("_config_dir", ".")).resolve()
                            validation_folder = (config_dir / auto_validation_path).resolve()

                            # Fixed script name is test_runner.py (generated during build via -v flag)
                            # It's located in the validation folder subdirectories (e.g., gdino/.tmp/test_runner.py)
                            # Determine which subdirectory based on TAO_MODEL_NAME
                            tao_model_name = test_config.get("env", {}).get("TAO_MODEL_NAME", "")
                            if not tao_model_name:
                                logger.error("❌ TAO_MODEL_NAME not specified in test_config.env")
                                client_rc = 1
                                client_stdout = ""
                                client_stderr = "TAO_MODEL_NAME not specified"
                            else:
                                validation_script_path = validation_folder / ".tmp" / "test_runner.py"

                                if not validation_script_path.exists():
                                    logger.error(f"❌ Validation script not found: {validation_script_path}")
                                    client_rc = 1
                                    client_stdout = ""
                                    client_stderr = f"Validation script not found: {validation_script_path}"
                                else:
                                    logger.info(f"🔧 Running validation script: {validation_script_path}")

                                    # Set up environment variables for validation script
                                    validation_env = os.environ.copy()
                                    if "env" in test_config:
                                        validation_env.update({k: str(v) for k, v in test_config["env"].items()})

                                    # Add service host for validation script to connect to server
                                    # The validation script expects TEST_HOST environment variable
                                    # Include port 8000 (the server's actual port, not from config)
                                    validation_env["TEST_HOST"] = f"http://{service_host}:8000"

                                    # Run validation script with Python
                                    validation_proc = subprocess.run(
                                        ["python", str(validation_script_path)],
                                        capture_output=True,
                                        text=True,
                                        timeout=max(60, timeout),
                                        cwd=str(validation_script_path.parent),
                                        env=validation_env
                                    )
                                    client_rc = validation_proc.returncode
                                    client_stdout = validation_proc.stdout
                                    client_stderr = validation_proc.stderr

                                    if client_rc == 0:
                                        logger.info("✅ Validation script completed successfully")
                                        if client_stdout:
                                            logger.info(f"Validation output:\n{client_stdout}")
                                    else:
                                        logger.error(f"❌ Validation script failed with return code {client_rc}")
                                        if client_stderr:
                                            logger.error(f"Validation stderr:\n{client_stderr}")
                                        if client_stdout:
                                            logger.error(f"Validation stdout:\n{client_stdout}")
                        except subprocess.TimeoutExpired:
                            logger.error("❌ Validation script timed out")
                            client_rc = 1
                            client_stdout = ""
                            client_stderr = "Validation script timed out"
                        except Exception as e:
                            logger.error(f"❌ Exception running validation script: {e}")
                            client_rc = 1
                            client_stdout = ""
                            client_stderr = str(e)
                    else:
                        logger.info("✅ Server is ready. Launching concurrent curl requests...")
                        # Read a single NDJSON file and launch curl for each line
                        # Get payloads_path from test_config, with a default fallback based on app name
                        test_app_name = test_config.get("TEST_APP_NAME") or None
                        default_payloads_path = f"{test_app_name}/payloads/payloads.jsonl"
                        payloads_path_str = test_config.get("payloads_path", default_payloads_path)
                        payloads_path = Path(payloads_path_str)

                        if not payloads_path.exists():
                            logger.error(f"❌ Payloads file not found: {payloads_path}")
                        else:
                            with payloads_path.open("r") as f:
                                payload_lines = [line.strip() for line in f if line.strip()]

                            procs: List[subprocess.Popen] = []
                            for line in payload_lines:
                                procs.append(subprocess.Popen(
                                    [
                                        "curl", "-sS", "-X", "POST", "-w",
                                        "%{http_code}",
                                        "-H", "Content-Type: application/json",
                                        "--data", line,
                                        f"http://{service_host}:8000/v1/inference"
                                    ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True
                                ))

                            outs: List[str] = []
                            errs: List[str] = []
                            client_rc = 0
                            http_status_errors = []
                            for i, p in enumerate(procs):
                                out, err = p.communicate(timeout=max(5, timeout))
                                outs.append(out or "")
                                errs.append(err or "")
                                if p.returncode != 0:
                                    client_rc = p.returncode
                                else:
                                    # For non-serverless, check HTTP status is 200
                                    # curl -w "%{http_code}" appends status to stdout
                                    if out and len(out) >= 3:
                                        # Extract last 3 chars as HTTP status code
                                        http_status = out[-3:]
                                        if http_status != "200":
                                            http_status_errors.append(
                                                f"Request {i+1}: HTTP {http_status}"
                                            )
                                            client_rc = 1  # Mark as failed

                            client_stdout = "\n".join(outs)
                            client_stderr = "\n".join(errs)

                            # Log HTTP status errors for non-serverless
                            if http_status_errors:
                                error_msg = (
                                    f"Non-serverless server returned non-200 status "
                                    f"codes: {'; '.join(http_status_errors)}"
                                )
                                logger.error("❌ %s", error_msg)
                                client_stderr += f"\n{error_msg}"

                # Always attempt to stop and collect logs
                logger.info("🛑 Stopping server container")

                # Collect logs BEFORE stopping container (while it's still running)
                logger.info("📋 Collecting container logs...")
                logs_proc = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
                server_logs = logs_proc.stdout
                if logs_proc.stderr:
                    server_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                # Now stop and remove container
                subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)

                # Prepare a Result-like object
                class Result:
                    def __init__(self, returncode, stdout, stderr):
                        self.returncode = returncode
                        self.stdout = stdout
                        self.stderr = stderr

                combined_stdout = "".join([
                    "=== SERVER LOGS ===\n", server_logs or "",
                    "\n=== CLIENT STDOUT ===\n", client_stdout or "",
                ])
                combined_stderr = client_stderr or ""
                result = Result(0 if (ready and client_rc == 0) else 1, combined_stdout, combined_stderr)

            # Save logs to file
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write(f"Error Detection: {'Enabled' if error_detection_enabled else 'Disabled'}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write("\n=== END LOG ===\n")

            # Check for ERROR patterns in the output only if error detection is enabled
            if error_detection_enabled:
                error_patterns = error_detection_config.get("patterns", [
                    "ERROR",
                    "Error",
                    "error",
                    "CRITICAL",
                    "Critical",
                    "critical",
                    "FATAL",
                    "Fatal",
                    "fatal",
                    "Exception:",
                    "exception:",
                    "Traceback",
                    "traceback"
                ])

                has_errors = False
                error_lines = []

                # Check stdout for errors
                for line in result.stdout.split('\n'):
                    for pattern in error_patterns:
                        if pattern in line:
                            has_errors = True
                            error_lines.append(f"STDOUT: {line.strip()}")
                            break

                # Check stderr for errors
                for line in result.stderr.split('\n'):
                    for pattern in error_patterns:
                        if pattern in line:
                            has_errors = True
                            error_lines.append(f"STDERR: {line.strip()}")
                            break

                if result.returncode == 0 and not has_errors:
                    logger.info(f"✅ Successfully tested image: {image_name}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return True, result.stdout, str(log_file)
                elif has_errors:
                    error_msg = (
                        f"Test failed due to ERROR logs detected in image: {image_name}"
                    )
                    logger.error(f"❌ {error_msg}")
                    logger.error("Error lines found:")
                    for error_line in error_lines:
                        logger.error(f"  {error_line}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, error_msg, str(log_file)
                else:
                    logger.warning(
                        f"⚠️  Test completed with non-zero return code for image: {image_name}"
                    )
                    logger.warning(f"Return code: {result.returncode}")
                    logger.warning(f"Output: {result.stdout}")
                    logger.warning(f"Error: {result.stderr}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, f"Container exited with code {result.returncode}", str(log_file)
            else:
                # When error detection is disabled, only check return code
                if result.returncode == 0:
                    logger.info(f"✅ Successfully tested image: {image_name}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return True, result.stdout, str(log_file)
                else:
                    logger.warning(
                        f"⚠️  Test completed with non-zero return code for image: {image_name}"
                    )
                    logger.warning(f"Return code: {result.returncode}")
                    logger.info(f"📄 Logs saved to: {log_file}")
                    return False, f"Container exited with code {result.returncode}", str(log_file)

        except subprocess.TimeoutExpired as timeout_ex:
            error_msg = f"Test timed out after {timeout} seconds for image: {image_name}"
            logger.error(f"❌ {error_msg}")

            # Try to collect container logs if this was a detached container
            container_logs = ""
            if server_type != "serverless":
                # For non-serverless, we have a detached container that needs cleanup
                try:
                    # Check if container_id was defined (container was started)
                    if 'container_id' in locals():
                        logger.info(f"📋 Collecting logs from timed-out container: {container_id}")
                        logs_proc = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        container_logs = logs_proc.stdout
                        if logs_proc.stderr:
                            container_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                        # Stop and remove the container
                        logger.info(f"🛑 Stopping timed-out container: {container_id}")
                        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True, timeout=30)
                        subprocess.run(["docker", "rm", container_id], capture_output=True, text=True, timeout=10)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to collect logs or cleanup container: {cleanup_error}")
                    container_logs += f"\n\n[Error during log collection: {cleanup_error}]\n"

            # Save timeout log with container logs if available
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Status: TIMEOUT\n")
                f.write(f"Timeout: {timeout} seconds\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                if container_logs:
                    f.write("\n\n=== CONTAINER LOGS ===\n")
                    f.write(container_logs)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)
        except Exception as e:
            error_msg = f"Exception during test: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Add full traceback for debugging
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Traceback:\n{full_traceback}")

            # Try to collect container logs if this was a detached container
            container_logs = ""
            if server_type != "serverless":
                # For non-serverless, we have a detached container that needs cleanup
                try:
                    # Check if container_id was defined (container was started)
                    if 'container_id' in locals():
                        logger.info(f"📋 Collecting logs from failed container: {container_id}")
                        logs_proc = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        container_logs = logs_proc.stdout
                        if logs_proc.stderr:
                            container_logs += "\n=== STDERR ===\n" + logs_proc.stderr

                        # Stop and remove the container
                        logger.info(f"🛑 Stopping failed container: {container_id}")
                        subprocess.run(["docker", "stop", container_id], capture_output=True, text=True, timeout=30)
                        subprocess.run(["docker", "rm", container_id], capture_output=True, text=True, timeout=10)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Failed to collect logs or cleanup container: {cleanup_error}")
                    container_logs += f"\n\n[Error during log collection: {cleanup_error}]\n"

            # Save exception log
            with open(log_file, 'w') as f:
                f.write("=== Test Configuration ===\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Status: EXCEPTION\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n=== ERROR ===\n")
                f.write(error_msg)
                f.write("\n\n=== TRACEBACK ===\n")
                f.write(full_traceback)
                if container_logs:
                    f.write("\n\n=== CONTAINER LOGS ===\n")
                    f.write(container_logs)
                f.write("\n=== END LOG ===\n")

            return False, error_msg, str(log_file)

    def cleanup_prerequisite_script(self, test_config: Dict, test_id: int) -> bool:
        """Clean up resources created by prerequisite scripts."""
        prerequisite_script = test_config.get("prerequisite_script")
        if not prerequisite_script:
            return True

        try:
            # Check if the script is the RTSP server setup script
            if "setup_rtsp_server.sh" in prerequisite_script:
                logger.info("🧹 Cleaning up RTSP server...")
                # Use secure command execution for cleanup
                cleanup_command = ["./setup_rtsp_server.sh", "--kill"]
                cleanup_result = subprocess.run(
                    cleanup_command,
                    shell=False,  # Safer execution without shell
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if cleanup_result.returncode == 0:
                    logger.info("✅ RTSP server cleanup completed")
                    return True
                else:
                    logger.warning(f"⚠️  RTSP server cleanup failed: {cleanup_result.stderr}")
                    return False

            # Add more cleanup logic for other prerequisite scripts here
            logger.info("🧹 No specific cleanup needed for prerequisite script")
            return True

        except Exception as e:
            logger.warning(f"⚠️  Exception during prerequisite cleanup: {str(e)}")
            return False

    def cleanup_image(self, image_name: str) -> bool:
        """Remove the test image."""
        try:
            result = subprocess.run(
                ["docker", "rmi", image_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"🧹 Cleaned up image: {image_name}")
                return True
            else:
                logger.warning(f"⚠️  Failed to cleanup image: {image_name}")
                return False

        except Exception as e:
            logger.warning(f"⚠️  Exception during cleanup: {str(e)}")
            return False

    def run_test_suite(self, test_configs: List[Dict], cleanup: bool = True, gitlab_token: Optional[str] = None, force_full_flow: bool = False) -> Dict:
        """Run a suite of tests with different configurations."""
        results = {
            "total_tests": len(test_configs),
            "passed": 0,
            "failed": 0,
            "results": []
        }

        for i, config in enumerate(test_configs, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test {i}/{len(test_configs)}")
            logger.info(f"{'='*60}")

            # Generate unique image name
            image_name = f"test-inference_builder-{i}-{int(time.time())}"

            build_args = config.get("build_args", {}).copy()
            test_cfg = config.get("test_config", {}).copy()

            # Add _config_dir to test_cfg so paths can be resolved correctly
            if "_config_dir" in config:
                test_cfg["_config_dir"] = config["_config_dir"]

            # Note: auto_validation is passed directly to build_image via test_config
            # No need to add it to build_args

            # Download models if specified in test config
            models_config = test_cfg.get("models", {})
            if models_config:
                logger.info(f"📦 Checking and downloading models...")
                config_dir = Path(config.get("_config_dir", "."))
                download_success, download_msg = self.download_models(models_config, config_dir)
                if not download_success:
                    logger.error(f"❌ Model download failed: {download_msg}")
                    results["failed"] += 1
                    test_result = {
                        "test_id": i,
                        "config": config,
                        "status": "FAILED",
                        "build_success": False,
                        "test_success": False,
                        "build_output": "",
                        "test_output": download_msg,
                        "log_file": "",
                        "image_name": image_name
                    }
                    results["results"].append(test_result)
                    logger.info(f"Test {i} result: FAILED (model download)")
                    continue

            # If default_enable is False, skip Docker build/test but run codegen,
            # unless full flow is forced by selection (e.g., --test-case provided)
            if not test_cfg.get("default_enable", True) and not force_full_flow:
                logger.info("⚙️  Test disabled via default_enable=false. Running code generation only...")
                codegen_success, codegen_output = self.generate_inference_code(build_args, test_cfg)

                status = "SKIPPED"
                test_result = {
                    "test_id": i,
                    "config": config,
                    "status": status,
                    "build_success": False,
                    "test_success": False,
                    "build_output": codegen_output,
                    "test_output": "",
                    "log_file": "",
                    "image_name": image_name
                }
                results["results"].append(test_result)
                logger.info(f"Test {i} result: {status} (codegen_only)")
                # Do not increment passed/failed counters for skipped tests
                continue

            # Build image with resolved dockerfile and base_dir
            dockerfile_to_use = config.get("_resolved_dockerfile")
            base_dir_to_use = config.get("_resolved_base_dir")
            build_success, build_output = self.build_image(build_args, image_name, dockerfile_to_use, base_dir_to_use, test_cfg)

            if build_success:
                # Test image - pass server type and config dir to test_config
                test_config = config.get("test_config", {}).copy()
                if "SERVER_TYPE" in build_args:
                    test_config["SERVER_TYPE"] = build_args["SERVER_TYPE"]
                # Pass config directory for relative path resolution
                if "_config_dir" in config:
                    test_config["_config_dir"] = config["_config_dir"]
                if "TEST_APP_NAME" in build_args:
                    test_config["TEST_APP_NAME"] = build_args["TEST_APP_NAME"]
                test_success, test_output, log_file = self.test_image(
                    image_name,
                    test_config,
                    i
                )

                if test_success:
                    results["passed"] += 1
                    status = "PASSED"
                else:
                    results["failed"] += 1
                    status = "FAILED"
            else:
                results["failed"] += 1
                test_success = False
                test_output = ""
                log_file = ""
                status = "FAILED"

            # Cleanup (run cleanup for full-flow tests; skip if codegen-only path was taken)
            if cleanup:
                self.cleanup_image(image_name)
                # Clean up prerequisite scripts
                self.cleanup_prerequisite_script(config.get("test_config", {}), i)

            # Store result
            test_result = {
                "test_id": i,
                "config": config,
                "status": status,
                "build_success": build_success,
                "test_success": test_success,
                "build_output": build_output,
                "test_output": test_output,
                "log_file": log_file,
                "image_name": image_name
            }

            results["results"].append(test_result)

            logger.info(f"Test {i} result: {status}")

        return results

    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """Generate a test report."""
        report = {
            "summary": {
                "total_tests": results["total_tests"],
                "passed": results["passed"],
                "failed": results["failed"],
                "success_rate": f"{(results['passed'] / results['total_tests'] * 100):.1f}%" if results["total_tests"] > 0 else "0%"
            },
            "results": results["results"]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to: {output_file}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total tests: {results['total_tests']}")
        logger.info(f"Passed: {results['passed']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success rate: {report['summary']['success_rate']}")

        # Print log file locations
        logger.info(f"\n{'='*60}")
        logger.info("LOG FILES")
        logger.info(f"{'='*60}")
        for result in results["results"]:
            if result.get("log_file"):
                logger.info(f"Test {result['test_id']}: {result['log_file']}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Test Docker builds with different arguments")
    parser.add_argument("--dockerfile", default="Dockerfile", help="Path to Dockerfile")
    parser.add_argument("--base-dir", default=".", help="Base directory for Docker build context")
    parser.add_argument("--config-file", required=True, help="JSON file with test configurations")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--log-dir", default="logs", help="Directory to save container logs")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup images after testing")
    parser.add_argument("--gitlab-token", help="GitLab token for authentication")
    parser.add_argument("--test-case", help="Run only the test case with this name (partial match supported). Supplying this forces full flow (build+test) even if disabled.")

    # Parse arguments with security validation
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"❌ Argument parsing failed: {str(e)}")
        sys.exit(1)

    # Comprehensive security validation
    validation_errors = []

    # Validate dockerfile path
    if not validate_dockerfile_path(args.dockerfile):
        validation_errors.append(f"Invalid Dockerfile path: {args.dockerfile}")

    # Validate base directory
    if not validate_safe_path(args.base_dir):
        validation_errors.append(f"Invalid base directory path: {args.base_dir}")

    # Validate config file
    if not validate_config_file_path(args.config_file):
        validation_errors.append(f"Invalid config file path: {args.config_file}")

    # Validate output file if provided
    if args.output and not validate_safe_path(args.output):
        validation_errors.append(f"Invalid output file path: {args.output}")

    # Validate log directory
    if not validate_log_directory(args.log_dir):
        validation_errors.append(f"Invalid log directory path: {args.log_dir}")

    # Validate GitLab token if provided
    if args.gitlab_token and not validate_gitlab_token(args.gitlab_token):
        validation_errors.append("Invalid GitLab token format")

    # Exit if any validation errors
    if validation_errors:
        logger.error("❌ Security validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Additional file existence checks with security
    try:
        # Validate config file first (required)
        if not os.path.isfile(args.config_file):
            logger.error(f"❌ Config file not found: {args.config_file}")
            sys.exit(1)

        # Dockerfile and base-dir are optional - they'll be resolved from config file if not found
        # Only warn if they're explicitly specified but don't exist
        dockerfile_exists = os.path.isfile(args.dockerfile)
        base_dir_exists = os.path.isdir(args.base_dir)

        if not dockerfile_exists:
            logger.info(f"ℹ️  Dockerfile not found at command-line path: {args.dockerfile}")
            logger.info(f"ℹ️  Will use Dockerfile from test config or config directory")

        if not base_dir_exists:
            logger.info(f"ℹ️  Base directory not found at command-line path: {args.base_dir}")
            logger.info(f"ℹ️  Will use base directory from test config or config directory")

        # Validate log directory path (prevent directory traversal)
        log_dir_path = Path(args.log_dir).resolve()
        current_dir = Path.cwd().resolve()
        try:
            log_dir_path.relative_to(current_dir)
        except ValueError:
            logger.error(f"❌ Log directory path is outside current directory: {args.log_dir}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ File validation failed: {str(e)}")
        sys.exit(1)

    # Initialize tester with validated arguments
    try:
        tester = DockerBuildTester(args.dockerfile, args.base_dir, args.log_dir)
    except Exception as e:
        logger.error(f"❌ Failed to initialize DockerBuildTester: {str(e)}")
        sys.exit(1)

    # Load test configurations from file with security validation
    try:
        with open(args.config_file, 'r') as f:
            test_configs = json.load(f)

        # Validate JSON structure
        if not isinstance(test_configs, list):
            logger.error("❌ Config file must contain a list of test configurations")
            sys.exit(1)

        # Get the directory containing the config file for resolving relative paths
        config_dir = Path(args.config_file).parent.resolve()
        logger.info(f"📁 Config file directory: {config_dir}")

        # Validate each test configuration for security
        for i, config in enumerate(test_configs):
            if not isinstance(config, dict):
                logger.error(f"❌ Test configuration {i+1} must be a dictionary")
                sys.exit(1)

            # Handle dockerfile specification
            # Priority: 1) dockerfile in test_config, 2) Dockerfile in config dir, 3) command-line arg
            if "dockerfile" in config:
                # User explicitly specified dockerfile in config
                dockerfile_path = (config_dir / config["dockerfile"]).resolve()
                if not dockerfile_path.exists():
                    logger.error(f"❌ Dockerfile specified in test config not found: {dockerfile_path}")
                    sys.exit(1)
                config["_resolved_dockerfile"] = str(dockerfile_path)
                logger.info(f"🔗 Using Dockerfile from config: {config['dockerfile']} -> {dockerfile_path}")
            else:
                # Check for Dockerfile in same directory as config
                default_dockerfile = config_dir / "Dockerfile"
                if default_dockerfile.exists():
                    config["_resolved_dockerfile"] = str(default_dockerfile)
                    logger.info(f"🔗 Using default Dockerfile from config directory: {default_dockerfile}")
                else:
                    # Fall back to command-line specified dockerfile
                    config["_resolved_dockerfile"] = args.dockerfile
                    logger.info(f"🔗 Using Dockerfile from command-line: {args.dockerfile}")

            # Base directory is ALWAYS the directory containing the test config JSON
            # This makes test configs self-contained and portable
            config["_resolved_base_dir"] = str(config_dir)
            config["_config_dir"] = str(config_dir)  # Store for model downloads
            logger.info(f"🔗 Using base_dir (config directory): {config_dir}")

            # Validate build_args if present
            if "build_args" in config:
                if not isinstance(config["build_args"], dict):
                    logger.error(f"❌ build_args in test configuration {i+1} must be a dictionary")
                    sys.exit(1)

                # Resolve paths relative to config file directory
                path_args = ["APP_YAML_PATH", "OUTPUT_DIR", "PROCESSORS_PATH", "OPENAPI_SPEC"]
                for path_arg in path_args:
                    if path_arg in config["build_args"]:
                        original_path = config["build_args"][path_arg]
                        # Resolve relative to config file directory
                        resolved_path = (config_dir / original_path).resolve()
                        # Convert back to relative path from current working directory
                        try:
                            rel_path = resolved_path.relative_to(Path.cwd().resolve())
                            config["build_args"][path_arg] = str(rel_path)
                            logger.info(f"🔗 Resolved {path_arg}: {original_path} -> {rel_path}")
                        except ValueError:
                            # If can't make relative, use absolute path
                            config["build_args"][path_arg] = str(resolved_path)
                            logger.info(f"🔗 Resolved {path_arg}: {original_path} -> {resolved_path}")

                for key, value in config["build_args"].items():
                    if not validate_build_arg_name(key):
                        logger.error(f"❌ Invalid build arg name in test configuration {i+1}: {key}")
                        sys.exit(1)
                    if not isinstance(value, str) or not validate_build_arg_value(value):
                        logger.error(f"❌ Invalid build arg value in test configuration {i+1}: {value}")
                        sys.exit(1)

            # Validate test_config if present
            if "test_config" in config:
                test_config = config["test_config"]
                config_valid, config_error = validate_test_config(test_config)
                if not config_valid:
                    logger.error(f"❌ Test configuration {i+1} validation failed: {config_error}")
                    sys.exit(1)

        # Filter test configurations by test case name if specified
        if args.test_case:
            original_count = len(test_configs)
            if args.test_case == "*":
                logger.info("--test-case '*' specified: selecting all tests (including disabled).")
            else:
                test_configs = [
                    config for config in test_configs
                    if args.test_case.lower() in config.get("name", "").lower()
                ]

                if not test_configs:
                    logger.error(f"❌ No test cases found matching '{args.test_case}'")
                    logger.info("Available test cases:")
                    # Reload original configs to show available names
                    with open(args.config_file, 'r') as f:
                        original_configs = json.load(f)
                    for i, config in enumerate(original_configs, 1):
                        logger.info(f"  {i}. {config.get('name', f'Unnamed test {i}')}")
                    sys.exit(1)

                logger.info(f"Filtered {original_count} test configurations to {len(test_configs)} matching '{args.test_case}'")
                for i, config in enumerate(test_configs, 1):
                    logger.info(f"  {i}. {config.get('name', f'Test {i}')}")
        else:
            # Include disabled tests; they will run codegen-only path unless -c/--test-case forces full flow.
            logger.info("Including disabled tests (default_enable=false). They will run codegen-only unless --test-case is provided.")

    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in config file: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load config file: {str(e)}")
        sys.exit(1)

    # Run tests with error handling
    try:
        logger.info(f"Starting test suite with {len(test_configs)} configurations")
        logger.info(f"Logs will be saved to: {args.log_dir}")
        # Force full flow if --test-case provided (including "*")
        force_full_flow = args.test_case is not None and len(args.test_case) > 0
        results = tester.run_test_suite(test_configs, cleanup=not args.no_cleanup, gitlab_token=args.gitlab_token, force_full_flow=force_full_flow)
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {str(e)}")
        sys.exit(1)

    # Generate report with error handling
    try:
        tester.generate_report(results, args.output)
    except Exception as e:
        logger.error(f"❌ Report generation failed: {str(e)}")
        sys.exit(1)

    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()