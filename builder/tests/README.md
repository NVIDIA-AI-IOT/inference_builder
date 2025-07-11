# Docker Container Test Suite

This directory contains a comprehensive test suite for testing Docker container builds with different arguments and configurations.

## Overview

The test suite allows you to:
- Test Docker builds with different TensorRT, CUDA, and cuDNN versions
- Run containers with various command line arguments
- Test different environment configurations
- Validate volume mounts and data access
- Generate detailed test reports
- **Capture and save container logs to files for debugging**
- **Automatically detect ERROR logs and fail tests when errors are found**

## Files

- `test_docker_builds.py` - Main test script
- `test_configs.json` - Test configuration file
- `run_tests.sh` - Shell script wrapper for easy usage
- `setup_test_data.sh` - Script to create test data directories
- `Dockerfile` - Dockerfile to test
- `README.md` - This documentation
- `logs/` - Directory containing container logs (created automatically)

## Quick Start

### 1. Setup Test Data

First, set up the test data directories:

```bash
chmod +x setup_test_data.sh
./setup_test_data.sh
```

### 2. Run Tests

#### Quick Test (Default Configuration)
```bash
chmod +x run_tests.sh
./run_tests.sh quick-test
```

#### Full Test Suite
```bash
./run_tests.sh full-test
```

#### Custom Configuration
```bash
./run_tests.sh custom-test --config my_config.json --log-dir custom_logs
```

## Test Configuration Structure

Each test configuration in `test_configs.json` has the following structure:

```json
{
  "name": "Test Name",
  "description": "Test description",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",

    "CACHE_BUSTER": "unique_value"
  },
  "test_config": {
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "/tmp/test_data": "/workspace/data"
    },
    "cmd": [
      "--input", "/workspace/data/video.mp4",
      "--output", "/workspace/data/output.mp4",
      "--batch-size", "4"
    ]
  }
}
```

### Build Arguments

- `TEST_APP_NAME`: Name of the test application
- `GITLAB_TOKEN`: **Required** - GitLab token for private repositories (use `${GITLAB_TOKEN}` placeholder)
- `TRT_VERSION_*`: TensorRT version components
- `CUDA_VERSION_*`: CUDA version components
- `CUDNN_VERSION`: cuDNN version
- `DS_TAO_APPS_TAG`: DS TAO apps git tag
- `CACHE_BUSTER`: Unique value to bypass Docker cache

### Test Configuration

- `timeout`: Timeout in seconds for the container test (default: 10 seconds)
- `env`: Environment variables to set in the container
- `volumes`: Volume mounts (host_path: container_path)
- `cmd`: Command line arguments to pass to the application
- `prerequisite_script`: Script to run before launching the docker container (optional)
  - Can be any shell command or script path
  - Useful for setting up test environments (e.g., starting RTSP servers)
  - Script logs are saved to `logs/prerequisite_{test_id}.log`
  - If script fails, the test is marked as failed
  - Automatic cleanup is performed after the test completes

## GITLAB_TOKEN Usage

The `GITLAB_TOKEN` is required for building images that access private GitLab repositories. Here's how to use it:

### In Test Configurations

Use the placeholder `${GITLAB_TOKEN}` in your test configurations:

```json
{
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "GITLAB_TOKEN": "${GITLAB_TOKEN}",
    "CACHE_BUSTER": "test1"
  }
}
```

### Providing the Token

**Option 1: Command Line (Recommended)**
```bash
./run_tests.sh quick-test --gitlab-token "your_actual_token_here"
```

**Option 2: Environment Variable**
```bash
export GITLAB_TOKEN="your_actual_token_here"
./run_tests.sh quick-test
```

**Option 3: Python Script Directly**
```bash
python3 test_docker_builds.py --gitlab-token "your_actual_token_here"
```

### Token Substitution

The test script automatically substitutes `${GITLAB_TOKEN}` with the actual token value:
- If token is provided: `${GITLAB_TOKEN}` → `your_actual_token_here`
- If token is not provided: The build argument is removed and a warning is logged

## Example Test Configurations

### 1. Basic Test
```json
{
  "name": "Basic Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "GITLAB_TOKEN": "${GITLAB_TOKEN}",
    "CACHE_BUSTER": "default_test"
  },
  "test_config": {
    "timeout": 10,
    "cmd": ["--help"]
  }
}
```

### 2. Video Stream Test
```json
{
  "name": "Video Stream Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "GITLAB_TOKEN": "${GITLAB_TOKEN}",
    "CACHE_BUSTER": "stream_test"
  },
  "test_config": {
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "cmd": [
      "--video-streams", "34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10"
    ]
  }
}
```

### 3. High Performance Mode
```json
{
  "name": "High Performance Mode",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "GITLAB_TOKEN": "${GITLAB_TOKEN}",
    "CACHE_BUSTER": "high_perf_test"
  },
  "test_config": {
    "timeout": 60,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes",
      "NVDS_ENABLE_LATENCY_MEASUREMENT": "1"
    },
    "volumes": {
      "/tmp/test_data": "/workspace/data"
    },
    "cmd": [
      "--input", "/workspace/data/sample_video.mp4",
      "--output", "/workspace/data/output.mp4",
      "--batch-size", "4",
      "--fps", "30",
      "--gpu-id", "0"
    ]
  }
}
```

### 4. RTSP Stream Test with Prerequisite Setup
```json
{
  "name": "RTSP Stream Test",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "GITLAB_TOKEN": "${GITLAB_TOKEN}",
    "CACHE_BUSTER": "rtsp_test"
  },
  "test_config": {
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "frame_sampling/models": "/workspace/models"
    },
    "cmd": [
      "--video-streams", "rtsp://localhost:8554/file-stream?frames=10"
    ],
    "prerequisite_script": "./setup_rtsp_server.sh sample_video.mp4 --daemon"
  }
}
```

## Command Line Arguments

The test script supports various command line arguments:

### Python Script (`test_docker_builds.py`)
```bash
python3 test_docker_builds.py [OPTIONS]

Options:
  --dockerfile PATH     Path to Dockerfile (default: Dockerfile)
  --base-dir PATH       Base directory for Docker build context (default: .)
  --config-file PATH    JSON file with test configurations (REQUIRED)
  --output PATH         Output file for test report
  --log-dir DIR         Directory to save container logs (default: logs)
  --no-cleanup          Don't cleanup images after testing
  --gitlab-token TOKEN  GitLab token for private repos
```

### Shell Script (`run_tests.sh`)
```bash
./run_tests.sh [COMMAND] [OPTIONS]

Commands:
  full-test      Run full test suite with all configurations from test_configs.json
  custom-test    Run test with custom configuration file
  help           Show help message

Options:
  --gitlab-token TOKEN    GitLab token for private repositories
  --no-cleanup           Don't cleanup Docker images after testing
  --output FILE          Save test report to specified file
  --config FILE          Use custom configuration file (required for custom-test)
  --log-dir DIR          Directory to save container logs (default: logs)
```

## Test Data Setup

The `setup_test_data.sh` script creates the following directory structure:

```
/tmp/
├── test_data/
│   ├── sample_video.mp4
│   ├── test_video.mp4
│   ├── multi_camera_feed.mp4
│   ├── custom_dataset.mp4
│   ├── rtsp_camera.txt
│   └── rtsp_output.txt
├── test_models/
│   └── ensemble_model.trt
├── custom_models/
│   ├── custom_model.trt
│   └── config.yaml
└── debug_output/
```

## Test Reports

Test reports are generated in JSON format and include:

- Summary statistics (total tests, passed, failed, success rate)
- Detailed results for each test
- Build and test outputs
- Configuration used for each test

Example report structure:
```json
{
  "summary": {
    "total_tests": 2,
    "passed": 2,
    "failed": 0,
    "success_rate": "100.0%"
  },
  "results": [
    {
      "test_id": 1,
      "name": "Default Configuration",
      "status": "PASSED",
      "build_success": true,
      "test_success": true,
      "build_output": "...",
      "test_output": "..."
    }
  ]
}
```

## Advanced Usage

### Custom Test Configurations

Create your own test configuration file:

```json
[
  {
    "name": "My Custom Test",
    "description": "Testing with custom parameters",
    "build_args": {
      "TEST_APP_NAME": "frame_sampling",
      "GITLAB_TOKEN": "${GITLAB_TOKEN}",
      "TRT_VERSION_MAJOR": "10",
      "TRT_VERSION_MINOR": "8",
      "CACHE_BUSTER": "my_test"
    },
    "test_config": {
      "timeout": 45,
      "env": {
        "CUSTOM_VAR": "custom_value"
      },
      "volumes": {
        "/path/to/my/data": "/workspace/data"
      },
      "cmd": [
        "--my-custom-arg", "value",
        "--another-arg", "another_value"
      ]
    }
  }
]
```

### Running Specific Tests

To run only specific tests, modify the configuration file or create a subset:

```bash
# Create a subset configuration
cat > my_tests.json << EOF
[
  $(head -n 20 test_configs.json | tail -n 15)
]
EOF

# Run with custom configuration
./run_tests.sh custom-test --config my_tests.json --gitlab-token "your_token"
```

### Continuous Integration

For CI/CD pipelines, you can run tests with specific configurations:

```bash
# Run tests and save report
./run_tests.sh full-test --output test_report.json --gitlab-token "$CI_GITLAB_TOKEN"

# Check exit code
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **GITLAB_TOKEN not provided**:
   - Error: `GITLAB_TOKEN placeholder found but no token provided`
   - Solution: Provide token via `--gitlab-token` option

2. **Docker not running**: Ensure Docker daemon is running

3. **Permission issues**: Run with appropriate permissions or use `sudo`

4. **Build failures**: Check if GitLab token has proper access to repositories

5. **Test timeouts**: Increase timeout in the script if needed

### Debug Mode

Run tests with verbose output:

```bash
# Enable debug logging
export PYTHONPATH=.
python3 -u test_docker_builds.py --config-file test_configs.json --no-cleanup --gitlab-token "your_token"
```

### Cleanup

Clean up test images manually if needed:

```bash
# List test images
docker images | grep test-inference-builder

# Remove all test images
docker images | grep test-inference-builder | awk '{print $3}' | xargs docker rmi
```

## Security Notes

- **Never commit GitLab tokens to version control**
- Use environment variables or CI/CD secrets for token storage
- Rotate tokens regularly
- Use tokens with minimal required permissions

## Contributing

To add new test configurations:

1. Add your configuration to `test_configs.json`
2. Include `"GITLAB_TOKEN": "${GITLAB_TOKEN}"` in build_args
3. Update this README if needed
4. Test your configuration locally
5. Submit a pull request

## License

This test suite is part of the inference-builder project and follows the same license terms.

## Log Dumping

The test suite automatically captures and saves container logs to files for debugging and analysis.

### Log File Structure

Each test run creates a log file with the following structure:
```
=== Test Configuration ===
Image: test-inference-builder-1-1234567890
Command: docker run --rm -e NVSTREAMMUX_ADAPTIVE_BATCHING=yes test-inference-builder-1-1234567890 --video-streams 34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10
Return Code: 0
Timestamp: 2024-01-15 10:30:45

=== STDOUT ===
[Application output here]

=== STDERR ===
[Error messages here]

=== END LOG ===
```

### Log File Naming

Log files are named using the pattern: `test_{test_id}_{image_name}.log`
- `test_id`: Sequential test number
- `image_name`: Docker image name (sanitized for filesystem)

### Log Directory

- **Default**: `logs/` directory in the current working directory
- **Custom**: Specify with `--log-dir` option
- **Auto-creation**: Directory is created automatically if it doesn't exist

### Example Usage

```bash
# Use default log directory
./run_tests.sh full-test

# Use custom log directory
./run_tests.sh full-test --log-dir /tmp/test_logs

# Python script directly
python3 test_docker_builds.py --log-dir my_logs --config-file test_configs.json
```

### Log File Locations in Reports

The test report includes the path to each log file:

```json
{
  "results": [
    {
      "test_id": 1,
      "log_file": "logs/test_1_test-inference-builder-1-1234567890.log",
      "status": "PASSED"
    }
  ]
}
```

## Error Detection

The test script includes built-in error detection that can be configured per test:

```json
{
  "error_detection": {
    "enabled": true,
    "patterns": [
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
    ]
  }
}
```

### Real-time Output Streaming

When `error_detection.enabled` is set to `false`, the container output is streamed to the host's stdout in real-time instead of being captured for analysis. This is useful for:

- **Interactive debugging**: See container output as it happens
- **Long-running processes**: Monitor progress without waiting for completion
- **Development workflows**: Get immediate feedback during testing

Example configuration for real-time output:
```json
{
  "name": "Real-time Output Test",
  "description": "Test with error detection disabled for real-time output streaming",
  "build_args": {
    "TEST_APP_NAME": "frame_sampling",
    "CACHE_BUSTER": "realtime_test"
  },
  "test_config": {
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "volumes": {
      "frame_sampling/models": "/workspace/models"
    },
    "cmd": [
      "--video-streams", "b69d7248-b351-4239-9819-4005e5375850?frames=10"
    ],
    "error_detection": {
      "enabled": false
    }
  }
}
```

**Behavior differences:**
- **Error detection enabled**: Output is captured and analyzed for error patterns. Test fails if errors are detected.
- **Error detection disabled**: Output is streamed to host stdout in real-time. Test only fails if container exits with non-zero return code.

**Note**: When error detection is disabled, logs are still saved to files for later review, but the output is also displayed in real-time on the host console.

## Volume Mounting

### Pre-build Commands

The test script automatically executes a pre-build command before each Docker build. This command is hardcoded and uses the `TEST_APP_NAME` from the build arguments to determine the sample folder:

**Hardcoded Command:**
```bash
python builder/main.py builder/tests/{TEST_APP_NAME}/app.yaml -o builder/tests/{TEST_APP_NAME} -c builder/tests/{TEST_APP_NAME}/processors.py --server-type serverless -t
```

**Example:**
- If `TEST_APP_NAME=frame_sampling`, the command becomes:
  ```bash
  python builder/main.py builder/tests/frame_sampling/app.yaml -o builder/tests/frame_sampling -c builder/tests/frame_sampling/processors.py --server-type serverless -t
  ```

This pre-build step is useful for:
- **Code generation**: Running scripts that generate code or configuration files
- **Dependency preparation**: Setting up dependencies or downloading assets
- **Environment setup**: Preparing the build environment
- **Validation**: Running tests or checks before building

**Pre-build Command Behavior:**
- **Automatic execution**: Runs before every Docker build in the test suite
- **Dynamic folder**: Uses `TEST_APP_NAME` from build arguments to determine the sample folder
- **Execution**: Runs in the current working directory before Docker build
- **Timeout**: 10-minute timeout (configurable in the script)
- **Failure handling**: If pre-build command fails, the test is marked as failed
- **Output**: Pre-build command output is logged and included in test results
- **Shell execution**: Commands are executed in a shell environment

### Timeout Configuration

Each test can specify a custom timeout value. If the container doesn't complete within the specified time, the test is marked as failed.

**Example:**
```json
{
  "test_config": {
    "timeout": 30,
    "env": {
      "NVSTREAMMUX_ADAPTIVE_BATCHING": "yes"
    },
    "cmd": [
      "--video-streams", "34888cef-8d7a-4de9-80f2-7a6a11974d6f?frames=10"
    ]
  }
}
```

**Timeout Behavior:**
- **Default**: 10 seconds if not specified
- **Failure**: Test fails if container doesn't complete within timeout
- **Logging**: Timeout information is logged and saved to log files
- **Flexible**: Different timeouts can be set for different tests