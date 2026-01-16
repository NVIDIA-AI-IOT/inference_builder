# Inference Builder MCP Integration

This document explains how to integrate the Inference Builder tool with Cursor via MCP (Model Context Protocol).

## Overview

The MCP integration provides a seamless way to use Inference Builder directly within Cursor, allowing you to:
- Generate inference pipelines from YAML configurations
- Explore sample configurations
- Build Docker images from generated pipelines
- All without leaving your development environment

## Installation

### 1. Install MCP Dependencies

```bash
pip install -r requirements-mcp.txt
```

### 2. Verify MCP Installation

```bash
python -c "import mcp; print('MCP installed successfully')"
```

## Available Tools

The MCP server provides the following tools:

### 1. `generate_inference_pipeline`

Generates an inference pipeline from a YAML configuration file.

**Parameters:**
- `config_file` (required): Path to the YAML configuration file
- `server_type` (optional): Type of server to generate (`triton`, `fastapi`, `nim`, `serverless`)
- `output_dir` (optional): Output directory for generated code
- `api_spec` (optional): Path to OpenAPI specification file
- `custom_modules` (optional): List of custom Python module files
- `exclude_lib` (optional): Exclude common library from generated code
- `tar_output` (optional): Create a tar.gz archive of the output
- `validation_dir` (optional): Validation directory path to build validator
- `no_docker` (optional): Use local OpenAPI Generator instead of Docker
- `test_cases_abs_path` (optional): Use absolute paths in generated test_cases.yaml

**Example:**
```
Generate a DeepStream object detection pipeline using the ds_detect.yaml configuration with serverless output.
```

### 2. `build_docker_image`

Builds a Docker image from a generated inference pipeline.

**Parameters:**
- `image_name` (required): Name for the Docker image
- `dockerfile` (required): Path to Dockerfile (must be placed in the output directory)

**Example:**
```
Build a Docker image from the generated deepstream-app pipeline with the name 'my-detector'.
```

### 3. `prepare_model_repository`

Prepares a model repository by downloading models from NGC or Hugging Face and copying configuration files.

**Parameters:**
- `model_configs` (required): List of model configurations to prepare
- `config_dir` (optional): Base directory for resolving relative config paths

**Example:**
```
Prepare a model repository for an NGC model with the path 'nvidia/tao/grounding_dino' and version 'v1.0'.
```

### 4. `docker_run_image`

Runs a Docker image with optional model repository mounting and environment configuration.

**Parameters:**
- `image_name` (required): Name of the Docker image to run
- `model_repo_host` (optional): Host path to model repository
- `model_repo_container` (optional): Container path for model repository (default: `/models`)
- `server_type` (optional): Server type hint (default: `serverless`)
- `env` (optional): Environment variables to set
- `cmd` (optional): Command-line arguments
- `gpus` (optional): GPU devices to use (default: `all`)
- `timeout` (optional): Timeout in seconds (default: 300)

**Example:**
```
Run the inference-app Docker image with the model repository from /path/to/models.
```

### 5. `generate_nvinfer_config`

Generates a DeepStream nvinfer runtime configuration file (nvdsinfer_config.yaml).

**Parameters:**
- `output_path` (required): Path where the generated config file should be saved
- `onnx_file` (required): Name of the ONNX model file
- `network_type` (required): Network type (0=detection, 1=classification, 2=segmentation, 3=instance_segmentation, 100=custom for raw tensor output)
- `input_dims` (required): Input dimensions in format 'channel;height;width' (e.g., '3;224;224')
- `label_file` (required): Name of the label file
- `precision_mode` (optional): Precision mode (0=FP32, 1=INT8, 2=FP16, default: 2)
- `custom_lib_path` (optional): Path to custom parser library
- `custom_parse_func` (optional): Symbol name of custom parsing function (required with custom_lib_path for detection/segmentation/instance_segmentation)
- `num_classes` (optional): Number of detected classes
- `gie_unique_id` (optional): Unique ID for this GIE (default: 1)
- `net_scale_factor` (optional): Scale factor for normalization (default: 0.00392156862745098)
- `offsets` (optional): Mean subtraction values in format 'R;G;B'
- `classifier_threshold` (optional): Confidence threshold for classification (default: 0.0)
- `input_tensor_from_meta` (optional): Whether to read input from metadata (0 or 1, default: 0)
- `output_tensor_meta` (optional): Whether to output raw tensor format (0 or 1, default: 0)

**Example:**
```
Generate a DeepStream nvinfer config for a detection model with ONNX file 'yolov5.onnx', input dimensions '3;640;640', and 80 classes.
```

**Important**: The `net_scale_factor` and `offsets` parameters must match your model's training preprocessing. See:
- [QUICK_NORMALIZATION_REFERENCE.md](QUICK_NORMALIZATION_REFERENCE.md) - Quick reference guide
- [STD_NORMALIZATION_CALCULATOR.md](STD_NORMALIZATION_CALCULATOR.md) - For models with std normalization

## Available Resources

The MCP server exposes resources for exploring schemas and samples:

### Schema Resources

- `schema://config.schema.json` - Main JSON Schema for configuration files
- `schema://readme` - Comprehensive schema documentation
- `schema://index.json` - **Schema navigation index** mapping backend types to their parameter schemas
- `schema://backends/{backend}.schema.json` - Backend-specific schemas (e.g., `deepstream`, `triton`, `vllm`, `tensorrtllm`, `pytorch`, `polygraphy`, `dummy`)
- `schema://backends/parameters/{backend}-parameters.schema.json` - Detailed parameter schemas for each backend

**Schema Navigation Flow:**
1. Read `schema://config.schema.json` for the main configuration structure
2. When a model specifies a `backend` type (e.g., `vllm`, `triton/python`), read `schema://index.json` to find the corresponding parameter schema path
3. Read the backend-specific parameter schema (e.g., `schema://backends/parameters/vllm-parameters.schema.json`) to get valid parameter options

### Sample Resources

Sample resources are dynamically discovered and categorized:

- `samples://config/{sample_path}` - Sample pipeline/application configuration YAML files
- `samples://runtime_config/{sample_path}` - DeepStream nvinfer runtime configuration files (nvdsinfer_config.yaml)
- `samples://runtime_preprocess/{sample_path}` - DeepStream nvdspreprocess runtime configuration files
- `samples://openapi/{sample_path}` - OpenAPI server specification YAML files
- `samples://dockerfile/{sample_path}` - Sample Dockerfiles
- `samples://processor/{sample_path}` - Sample preprocessor/postprocessor Python modules

**Example usage:**
```
Read the samples://config/ds_app/detection/ds_detect.yaml resource to see the DeepStream detection configuration.
Read the samples://runtime_config/ds_app/classification/changenet-classify/nvdsinfer_config.yaml resource to see a sample nvinfer runtime config.
```

## Configuration

### Cursor MCP Configuration

To use the MCP server with Cursor, you need to configure it in your Cursor settings:

1. **Open Cursor Settings**
2. **Navigate to AI Assistant or MCP settings**
3. **Add the MCP server configuration**

The exact configuration method depends on your Cursor version, but generally involves specifying:

```json
{
  "mcpServers": {
    "deepstream-inference-builder": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/inference-builder"
    }
  }
}
```

### Alternative: Manual MCP Server

You can also run the MCP server manually:

```bash
cd /path/to/inference-builder
python mcp_server.py
```

## Usage Examples

### Generate a DeepStream Pipeline

```
Use the generate_inference_pipeline tool to create a DeepStream object detection pipeline using the ds_detect.yaml configuration with serverless output.
```

### Explore Available Samples

```
List the available MCP resources to see sample configurations, Dockerfiles, and processors.
```

### View Sample Configurations

```
Read the samples://config/ds_app/detection/ds_detect.yaml resource to see the DeepStream detection configuration.
```

### Build Docker Image

```
Build a Docker image from the generated deepstream-app pipeline with the name 'my-detector' using the build_docker_image tool.
```

## Workflow Integration

### Typical Development Workflow

1. **Explore Samples**: Browse `samples://config/*` resources to see available configurations
2. **Examine Configurations**: Read sample resources to understand configuration patterns
3. **Create Your Config**: Modify a sample configuration or create your own
4. **Generate Pipeline**: Use `generate_inference_pipeline` to create your code
5. **Build Container**: Use `build_docker_image` to create a deployable container

### Integration with Version Control

The generated code can be:
- Committed to version control
- Used as a starting point for custom modifications
- Shared with team members
- Deployed to production environments

## Troubleshooting

### Common Issues

1. **MCP Server Not Found**
   - Ensure the MCP server is running
   - Check that the path in Cursor configuration is correct
   - Verify Python environment has required dependencies

2. **Configuration Validation Errors**
   - Check that required fields are present in your YAML
   - Ensure model definitions are complete
   - Verify backend specifications are correct

3. **Pipeline Generation Failures**
   - Check that the configuration file path is correct
   - Ensure all required dependencies are installed
   - Verify that the output directory is writable

### Debug Mode

To run the MCP server with debug information:

```bash
python -u mcp_server.py
```

## Advanced Features

### Custom Processors

The MCP integration supports custom preprocessors and postprocessors:

1. Create your custom processor classes
2. Reference them in your configuration file
3. Use the `custom_modules` parameter when generating pipelines

### Multi-Model Pipelines

Complex inference flows with multiple models are supported:

1. Define multiple models in your configuration
2. Specify routing between models
3. Use the `generate_inference_pipeline` tool to create the complete pipeline

### Server Types

Different server implementations are available:

- **FastAPI**: RESTful API server with automatic OpenAPI generation
- **Triton**: NVIDIA Triton Inference Server integration
- **NIM**: NVIDIA NIM server integration
- **Serverless**: Standalone command-line application

## Contributing

To extend the MCP integration:

1. Add new tools to the `InferenceBuilderMCPServer` class
2. Update the tool schemas in `list_tools`
3. Implement the tool logic in `call_tool`
4. Add tests for new functionality
5. Update this documentation

## Support

For issues with the MCP integration:

1. Check the troubleshooting section above
2. Verify your configuration and dependencies
3. Check the Inference Builder main documentation
4. Open an issue in the project repository

## License

This MCP integration follows the same license as the Inference Builder project (Apache 2.0).

