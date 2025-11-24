# Inference Builder JSON Schemas

This directory contains JSON Schema definitions for NVIDIA Inference Builder YAML configurations. These schemas provide validation, autocompletion, and documentation for creating inference pipeline configurations.

## Overview

The Inference Builder allows you to create inference pipelines with various backends such as DeepStream, TensorRT, Triton, TensorRT-LLM, vLLM, etc. These schemas help validate your configuration files and provide IDE support.

## Schema Organization

```
schemas/
├── config.schema.json              # Main configuration schema
├── backends/                       # Backend-specific schemas
│   ├── deepstream.schema.json     # DeepStream backend
│   ├── triton.schema.json         # Triton backend
│   ├── vllm.schema.json           # vLLM backend
│   ├── tensorrtllm.schema.json    # TensorRT-LLM backend
│   ├── polygraphy.schema.json     # Polygraphy backend
│   ├── dummy.schema.json          # Dummy backend
│   └── parameters/                # Backend-specific parameters
│       ├── deepstream-parameters.schema.json
│       ├── triton-parameters.schema.json
│       ├── vllm-parameters.schema.json
│       ├── tensorrtllm-parameters.schema.json
│       ├── polygraphy-parameters.schema.json
│       └── dummy-parameters.schema.json
├── common/                         # Common definitions
│   ├── base-model.schema.json     # Base model schema (shared structure)
│   ├── definitions.schema.json    # Shared type definitions
│   ├── preprocessors.schema.json  # Preprocessor schemas
│   └── postprocessors.schema.json # Postprocessor schemas
└── README.md                       # This file
```

### Schema Composition

The schemas use JSON Schema's composition features for better maintainability:

- **Base Model Schema** (`common/base-model.schema.json`): Defines the common structure shared by all backends (name, backend, input, output, max_batch_size, preprocessors, postprocessors)
- **Backend Schemas** (`backends/*.schema.json`): Extend the base model using `allOf` and specify which backend types are valid
- **Parameter Schemas** (`backends/parameters/*.schema.json`): Define backend-specific parameters with validation rules

This approach:
- ✅ Eliminates duplication across backend schemas
- ✅ Ensures consistency in common fields
- ✅ Makes it easy to add new backends
- ✅ Provides clear validation for backend-specific options

## Main Configuration Schema

The main schema (`config.schema.json`) defines the structure of inference builder configuration files:

### Required Fields

- `name`: String identifier for the microservice
- `model_repo`: Path to the directory containing model files
- `models`: Array of model specifications (at least one required)

### Optional Fields

- `input`: Array of input tensor specifications (required when pipeline includes multiple models or when inputs use custom types)
- `output`: Array of output tensor specifications (required when pipeline includes multiple models or when outputs use custom types)
- `server`: Server configuration including responders (not required for serverless)
- `routes`: Route map for tensor flow between models (required for multi-model pipelines)
- `postprocessors`: Top-level postprocessors (for consolidating outputs from multiple models)

### Example Configuration

```yaml
name: "my_service"
model_repo: "/workspace/model-repo"

input:
  - name: "text"
    data_type: TYPE_STRING
    dims: [-1]
  - name: "images"
    data_type: TYPE_CUSTOM_IMAGE_BASE64
    dims: [-1]
    optional: true

output:
  - name: "output"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]

models:
  - name: "my_model"
    backend: "triton/tensorrt"
    max_batch_size: 1
    input:
      - name: "input0"
        data_type: TYPE_FP32
        dims: [3, 768, 768]
    output:
      - name: "output0"
        data_type: TYPE_FP32
        dims: [10, 768, 768]
```

## Backend-Specific Schemas

### DeepStream Backend

**Files**:
- Schema: `backends/deepstream.schema.json`
- Parameters: `backends/parameters/deepstream-parameters.schema.json`

Used for DeepStream-based inference pipelines.

**Backend Type**: `deepstream/nvinfer`

**Parameters** (defined in `deepstream-parameters.schema.json`):
- `infer_config_path`: Array of paths to nvdsinfer_config.yaml files (required)
- `preprocess_config_path`: Array of paths to nvdspreprocess configuration files (optional)
- `batch_timeout`: Timeout in microseconds for batching multiple inputs
- `inference_timeout`: Inference timeout in seconds
- `tracker_config`: Configuration for object tracking
- `msgbroker_config`: Configuration for message broker integration
- `render_config`: Configuration for rendering and display
- `perf_config`: Performance monitoring configuration
- `kitti_output_path`: Paths for KITTI format output dumps

**Example**:
```yaml
models:
  - name: tao
    backend: deepstream/nvinfer
    max_batch_size: 1
    parameters:
      infer_config_path:
        - nvdsinfer_config.yaml
      preprocess_config_path:
        - nvdspreprocess_config.yaml
      batch_timeout: -1
      inference_timeout: 5
```

### Triton Backend

**Files**:
- Schema: `backends/triton.schema.json`
- Parameters: `backends/parameters/triton-parameters.schema.json`

Used for Triton Inference Server with various frameworks.

**Backend Types**:
- `triton/python`
- `triton/tensorrt`
- `triton/onnx`
- `triton/pytorch`

**Parameters**:
Triton backend parameters are intentionally open and backend-specific. All parameters are converted to `string_value` in the generated `config.pbtxt` file as defined in Triton's ModelConfig protobuf schema ([model_config.proto](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)).

Common parameters include:
- `FORCE_CPU_ONLY_INPUT_TENSORS`: Controls whether input tensors are kept on GPU or copied to CPU (Triton Python backend). Set to `"no"` to keep tensors on GPU for better performance.
- Backend-specific parameters (e.g., TensorRT engine paths, ONNX model paths, etc.)

For complete parameter documentation, refer to the [Triton Model Configuration documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_configuration.html).

**Example**:
```yaml
models:
  - name: visual_changenet
    backend: "triton/tensorrt"
    max_batch_size: 1
    parameters:
      FORCE_CPU_ONLY_INPUT_TENSORS: "no"
```

### vLLM Backend

**File**: `backends/vllm.schema.json`

Used for vLLM-based large language model inference.

**Backend Type**: `vllm`

**Key Parameters**:
- `max_num_tokens`: Maximum number of tokens
- `async_mode`: Enable asynchronous mode
- `gpu_memory_utilization`: GPU memory utilization (0.0 to 1.0)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism

**Example**:
```yaml
models:
  - name: "Cosmos"
    backend: "vllm"
    parameters:
      max_num_tokens: 19200
      async_mode: true
```

### TensorRT-LLM Backend

**File**: `backends/tensorrtllm.schema.json`

Used for TensorRT-LLM inference with optional PyTorch fallback.

**Backend Types**:
- `tensorrtllm`
- `tensorrtllm/pytorch`

**Key Parameters**:
- `max_num_tokens`: Maximum number of tokens
- `kv_cache_config`: KV cache configuration
  - `enable_block_reuse`: Enable KV cache block reuse
- `decoding_config`: Decoding configuration

**Example**:
```yaml
models:
  - name: "Qwen2.5-VL"
    backend: "tensorrtllm/pytorch"
    parameters:
      max_num_tokens: 19200
      kv_cache_config:
        enable_block_reuse: false
```

### Polygraphy Backend

**File**: `backends/polygraphy.schema.json`

Used for TensorRT inference via Polygraphy.

**Backend Type**: `polygraphy`

**Key Parameters**:
- `tensorrt_engine`: Path to TensorRT engine file (required)
- `precision`: Inference precision (fp32, fp16, int8)

**Example**:
```yaml
models:
  - name: nvclip_text
    backend: "polygraphy"
    max_batch_size: 64
    parameters:
      tensorrt_engine: "model.plan"
```

## Data Types

The following data types are supported:

### Numeric Types
- `TYPE_BOOL`, `TYPE_UINT8`, `TYPE_UINT16`, `TYPE_UINT32`, `TYPE_UINT64`
- `TYPE_INT8`, `TYPE_INT16`, `TYPE_INT32`, `TYPE_INT64`
- `TYPE_FP16`, `TYPE_FP32`, `TYPE_FP64`

### Basic Types
- `TYPE_STRING`: String data

### Custom Types
- `TYPE_CUSTOM_IMAGE_BASE64`: Base64-encoded images. When used for an input, the data will be decoded into an image tensors by default. When used for an output, the image tensors are encoded into a base64 string.
- `TYPE_CUSTOM_IMAGE_ASSETS`: Video asset strings. When used for an input, the is treated as image assets references. The assets will be automatically passed to the image decoder and converted into image tensors.
- `TYPE_CUSTOM_VIDEO_ASSETS`: Video asset strings. When used for an input, the data is interpreted as video assets with parameters controlling frame sampling. The assets are automatically decoded and evenly sampled into image tensors, which are then passed downstream frame by frame. The frame sampling parameters are provided as a query string in the format: ?key1=value1&key2=value2. Supported keys include:
    - frames: Number of total frames to be extracted.
    - start: Start timestamp in nanoseconds: 10*1e9 for 10 seconds.
    - duration: Duration in nanoseconds: 10*1e9 for 10 seconds.
- `TYPE_CUSTOM_VIDEO_CHUNK_ASSETS`: Video chunk asset strings. When used for an input, the data is interpreted as video assets with parameters controlling frame sampling and chunking. Assets are automatically decoded, chunked, and evenly sampled into image-tensor batches. Each chunk carries a stack of frames in image-tensor format. Frame-sampling parameters are specified in the query string: ?key1=value1&key2=value2. Supported keys include:
    - chunks: Number of chunks to split the video.
    - frames: Number of total frames to be extracted per chunk.
    - start: Start timestamp in nanoseconds: 10*1e9 for 10 seconds.
    - duration: Duration in nanoseconds: 10*1e9 for 10 seconds.
- `TYPE_CUSTOM_BINARY_BASE64`: Binary data encoded in base64. When used for an input, the data is automatically decoded into uint8 tensors. For output data, the data will be base64 encoded to a string.
- `TYPE_CUSTOM_BINARY_URLS`: When used for input, the data will be converted to a list and treated as urls.
- `TYPE_CUSTOM_VLM_INPUT`: Vision-Language Model input objects.
- `TYPE_CUSTOM_OBJECT`: Generic object type.

### DeepStream-Specific Types
- `TYPE_CUSTOM_DS_IMAGE`: 1-D uint8 tensor for DeepStream image data
- `TYPE_CUSTOM_DS_METADATA`: DeepStream metadata object
- `TYPE_CUSTOM_DS_MIME`: String for DeepStream MIME type
- `TYPE_CUSTOM_DS_SOURCE_CONFIG`: String for DeepStream source configuration path

## Preprocessors and Postprocessors

### Preprocessors

**File**: `common/preprocessors.schema.json`

Preprocessors transform input data before model inference.

**Common Preprocessors**:

1. **Image Preprocessors**
   - `changenet-normalizer`: Normalize images for ChangeNet
   - `nvclip-vision-preprocessor`: Preprocess images for NVCLIP
   - Config: `network_size`, `mean`, `std`

2. **Tokenizer Preprocessors**
   - `openclip-tokenizer`: Tokenize text for CLIP
   - `dummy-tokenizer`: Test tokenizer
   - Config: `max_length`, `truncation`, `padding`

3. **VLM Loaders**
   - `qwen-vl-loader`: Load inputs for Qwen VL models
   - `qwen-vl-image-loader`: Load images for Qwen VL
   - `qwen-vl-video-loader`: Load videos for Qwen VL
   - Config: `num_frames`, `fps`, `max_pixels`, `min_pixels`

**Example**:
```yaml
preprocessors:
  - kind: "custom"
    name: "changenet-normalizer"
    input: ["reference_image", "test_image"]
    output: ["input0", "input1"]
    config:
      network_size: [768, 768]
```

### Postprocessors

**File**: `common/postprocessors.schema.json`

Postprocessors transform model outputs.

**Common Postprocessors**:

1. **Masking Postprocessor**
   - `changenet-masking`: Apply masking to ChangeNet outputs
   - Config: `network_size`, `n_class`, `threshold`

2. **Embedding Postprocessor**
   - `nvclip-postprocessor`: Process NVCLIP embeddings
   - Config: `normalize`, `embedding_dim`

**Example**:
```yaml
postprocessors:
  - kind: "custom"
    name: "changenet-masking"
    input: ["output_final"]
    output: ["output_final"]
    config:
      network_size: [768, 768]
      n_class: 10
```

## Server Configuration

The server section defines how the inference service handles requests and responses.

### Responders

Responders map API operations to request/response transformations using Jinja2 templates.

**Available Responder Keys** (must match templates in `templates/responder/`):
- `infer` - Main inference endpoint
- `add_file` - Upload media file
- `del_file` - Delete media file
- `list_files` - List media files
- `add_live_stream` - Add live stream
- `del_live_stream` - Delete live stream
- `list_live_streams` - List live streams
- `healthy_ready` - Health check

**Common Operations** (values for the `operation` field):
- `inference`: General inference operation
- `create_chat_completion_v1_chat_completions_post`: Chat completion for LLMs
- `create_embedding`: Embedding generation
- `add_media_file`, `delete_media_file`, `list_media_files`: Media file management
- `add_live_stream`, `delete_live_stream`, `list_live_streams`: Live stream management
- `health_ready_v1_health_ready_get`: Health check

See [RESPONDERS.md](RESPONDERS.md) for complete documentation.

**Example**:
```yaml
server:
  responders:
    infer:
      operation: inference
      requests:
        InferenceRequest: >
          {
            "images": [{{ request.input[0]|tojson }}]
          }
      responses:
        InferenceResponse: >
          {
            "data": {{ response.output|tojson }},
            "model": "my-model"
          }
```

## Routes

Routes define the flow of tensors between models in multi-model pipelines.

**Format**: `<source>:<tensor_list> : <destination>:<tensor_list>`

- Source/destination can be model names or `:` for top-level
- Tensor list is optional: `["tensor1", "tensor2"]`

**Example**:
```yaml
routes:
  ':["reference_image", "test_image"]': 'visual_changenet'
  'visual_changenet:["output_final"]': ':["output"]'
```

## Using the Schemas

### In VS Code

Add to your workspace settings (`.vscode/settings.json`):

```json
{
  "yaml.schemas": {
    "./schemas/config.schema.json": "*.yaml"
  }
}
```

Or use the GitHub URL for the latest schema:

```json
{
  "yaml.schemas": {
    "https://raw.githubusercontent.com/NVIDIA-AI-IOT/inference_builder/main/schemas/config.schema.json": "*.yaml"
  }
}
```

### With YAML Language Server

Add to your `.yaml-language-server.json`:

```json
{
  "yaml.schemas": {
    "file:///path/to/schemas/config.schema.json": "/*.yaml"
  }
}
```

### Validation with Python

```python
import json
import yaml
from jsonschema import validate

# Load schema
with open('schemas/config.schema.json') as f:
    schema = json.load(f)

# Load and validate config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

validate(instance=config, schema=schema)
```

## Examples

See the `builder/samples/` directory for complete examples of configurations for different backends:

- `builder/samples/dummy/dummy.yaml` - Dummy backend example
- `builder/samples/vllm/vllm_cosmos.yaml` - vLLM backend example
- `builder/samples/qwen/trtllm_qwen.yaml` - TensorRT-LLM backend example
- `builder/samples/tao/ds_tao.yaml` - DeepStream backend example
- `builder/samples/changenet/trt_changenet.yaml` - Triton/TensorRT backend example
- `builder/samples/nvclip/tensorrt_nvclip.yaml` - Polygraphy backend example

## References

- [Inference Builder Documentation](../README.md)
- [JSON Schema Specification](https://json-schema.org/)
- [YAML Language Server](https://github.com/redhat-developer/yaml-language-server)

## Contributing

When adding new backends or features:

1. Update the appropriate schema files
2. Add examples to this README
3. Test with sample configurations
4. Update the `common/definitions.schema.json` if adding new types

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

