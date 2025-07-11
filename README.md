# Inference Builder

## Overview

Inference Builder is a tool that automatically generates inference pipelines and integrates them into either a microservice or a standalone application. It takes an inference configuration file and an OpenAPI specification (when integrated with an HTTP server) as inputs, and may also require custom code snippets in certain cases.

The output of the tool is a Python package that can be used to build a microservice container image with a customized Dockerfile.

The Inference Builder consists of three major components:

- Code templates: These are reusable modules for various inference backends and frameworks, as well as for API servers. They are optimized and tested, making them suitable for any model with specified inputs, outputs, and configuration parameters.
- Common inference flow: It serves as the core logic that standardizes the end-to-end inference process—including data loading and pre-processing, model inference, post-processing, and integration with the API server. It supports pluggable inference backends and frameworks, enabling flexibility and performance optimization."\
- Command line tool: It generates a source code package by combining predefined code templates with the Common Inference Flow. It also automatically produces corresponding test cases and evaluation scripts to support validation and performance assessment.

## Benefit of using Inference Builder

Compared to manually crafting inference source code, Inference Builder offers developers the following advantages:
- Separation of concerns: Introduces a new programming paradigm that decouples inference data flow and server logic from the model implementation, allowing developers to focus solely on model behavior.
- Backend flexibility: Standardizes data flow across different inference backends, enabling developers to switch to the optimal backend for each use case without rewriting the entire pipeline.
- Hardware acceleration: Automatically enables GPU-accelerated hardware codecs to boost performance.
- Streaming support: Provides built-in support for streaming protocols such as RTSP with minimal configuration.
- Standardized testing: Automates and standardizes test case generation to simplify validation and evaluation workflows.

## Getting started

### Clone the repository

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/DeepStreamSDK/inference-builder.git
cd inference-builder
git submodule update --init --recursive
```

### Install prerequisites

```bash
sudo apt update
sudo apt install protobuf-compiler
```

Create python virtual env (Optional) and install dependent packages

```bash
$ python -m venv /path/to/new/virtual/environment
# Activate the venc
$ source /path/to/new/virtual/environment/bin/activate
# Install the dependent packages
$ cd inference-builder
$ pip3 install -r requirements.txt

```

Ensure nvidia runtime added to `/etc/docker/daemon.json` to run GPU-enabled containers

```bash
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

## Usage

Before using the tool, you must prepare a YAML configuration file to define the inference flow. If server integration is required, you also need to provide an OpenAPI specification that defines the server and update the configuration with server templates based on the OpenAPI specification.

For generating the inference code with the corresponding server implementation, you can run the following command:

```bash
python builder/main.py -h
usage: Inference Builder [-h] [--server-type [{triton,fastapi,nim,serverless}]] [-o [OUTPUT_DIR]] [-a [API_SPEC]] [-c [CUSTOM_MODULE ...]] [-x] [-t] [-v VALIDATION_DIR] [--no-docker] [--test-cases-abs-path] config

positional arguments:
  config                Path the the configuration

options:
  -h, --help            show this help message and exit
  --server-type [{triton,fastapi,nim,serverless}]
                        Choose the server type
  -o [OUTPUT_DIR], --output-dir [OUTPUT_DIR]
                        Output directory
  -a [API_SPEC], --api-spec [API_SPEC]
                        File for OpenAPI specification
  -c [CUSTOM_MODULE ...], --custom-module [CUSTOM_MODULE ...]
                        Custom python modules
  -x, --exclude-lib     Don't include common lib to the generated code.
  -t, --tar-output      Zip the output to a single file
  -v VALIDATION_DIR, --validation-dir VALIDATION_DIR
                        valid validation directory path to build validator
  --no-docker           Use local OpenAPI Generator instead of Docker for OpenAPI client generation
  --test-cases-abs-path
                        Use absolute paths in generated test_cases.yaml
```

You can start with the builtin samples under _samples_ folder for generating inference pipeline and server implementation with various inference backends and frameworks.

## Configuration File

The configuration file is a YAML file that defines the inference flow and server implementation. It contains the following sections:

- name: The name of the inference pipeline, which will be used as the name of the folder to save the generated inference code or the name of the tarball if '-t' is specified.
- model_repo: Specifies the path to the model repository from which the inference pipeline searches and loads the required models.
- models: List of model definitions. A inference flow can incoperate multiple models, each might be implemented with different backends to achieve optimal performance.
- input(optional): Defines the top-level inputs of the inference flow. This field is required only when the pipeline includes multiple models or when the input type is not natively supported by the model backend—such as in cases that require standard preprocessing like video decoding.
- output(optional): Defines the top-level outputs of the inference flow. This field is required only when the pipeline includes multiple models or when the output type is not natively supported by the model backend—such as in cases that require standard preprocessing like video encoding.
- server(optional): Defines the endpoint templates for the server implementation. This field is not required if the server type is set to "serverless".
- routes(optional): Defines the routing rules for the inference flow when multiple models are involved.
- postprocessors(optional): Defines the top-level postprocessors for the inference flow. This field is required only when the pipeline includes multiple models and their output of these modols need to be consolidated.

A configuration file can be as simple as the below example:

```yaml

name: "detector"
model_repo: "/workspace/models"

models:
- name: rtdetr
  backend: deepstream/nvinfer
  max_batch_size: 4
  input:
  - name: media_url
    data_type: TYPE_CUSTOM_BINARY_URLS
    dims: [ -1 ]
  - name: mime
    data_type: TYPE_CUSTOM_DS_MIME
    dims: [ -1 ]
  output:
  - name: output
    data_type: TYPE_CUSTOM_DS_METADATA
    dims: [ -1 ]
  parameters:
    infer_config_path:
      - nvdsinfer_config.yaml
```

With the above configuration file, you can generate inference code for the RT-DETR object detection model using the DeepStream backend, which takes image or video url as input and produces bounding boxes as output. (Step by step guide can be found in the [detection README](./builder/samples/ds_app/detection/README.md))

Breakdown of the configuration:

- The inference package is named detector, and the model search path is set to "/workspace/models". The pipeline uses one model and will look for the model file (in this case, an ONNX file) under the "rtdetr" directory from the search path, as specified by the model name.
- Inference is performed using nvinfer from the NVIDIA DeepStream SDK. A corresponding configuration file named "nvdsinfer_config.yaml" must be present under "/workspace/models/rtdetr".
- The pipeline supports two inputs:
  - media_url: the path or URL to the input media.
  - mime: the media type (e.g., "video/mp4" or "image/jpeg").
- The pipeline supports batch processing with up to 4 media items at a time.
- The output is a custom DeepStream metadata object, which contains information about the detected bounding boxes.

### Input and Output Definition

Input and output definitions are required at the model level, and in some cases at the top level of the pipeline. Each input and output definition typically includes the following sections:

- name: a string representation to identify the data that passes through the input or output.
- data_type: Expected type of the data. The type is in string format and is derived from [basic data types defined by Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes) with following extensions:
  - TYPE_CUSTOM_IMAGE_BASE64: When used for an input, this type forces the data to be decoded from a base64-encoded string into an image tensor. When used for an output, the image tensor is encoded into a base64 string.
  - TYPE_CUSTOM_IMAGE_ASSETS: When used for an input, the data consists of UUID strings that reference managed image assets. These assets will be automatically passed to the image decoder and converted into image tensors.
  - TYPE_CUSTOM_VIDEO_ASSETS: When used as an input, the data consists of UUID strings that reference managed video assets, along with parameters that control frame sampling. These assets are automatically decoded to evenly sampled image tensors and passed downstream frame by frame. The frame sampling parameters are provided as a query string in the format: ?key1=value1&key2=value2. Supported keys include:
    - frames: Number of total frames to be extracted.
    - start: Start timestamp in microseconds
    - end: End timestamp in microseconds
  - TYPE_CUSTOM_VIDEO_CHUNK_ASSETS: similar to TYPE_CUSTOM_VIDEO_ASSETS, the data represents managed video assets and will be automatically decoded and sampled. The only difference is all the frames will be packaged into a single list before being passed downstream.
  - TYPE_BINARY_URLS: When used for input, the data will be converted to a list and treated as urls.
  - TYPE_CUSTOM_DS_IMAGE: Encoded image specifically for inputs of Deepstream pipeline
  - TYPE_CUSTOM_DS_MIME: Mime type used by Deepstream pipeline to determine the input media type
  - TYPE_CUSTOM_DS_METADATA: Structured output data specifically for Deepstream pipeline
- dims: The dimensions of the input or output in the form of a list. Each item in the list specifies the maximum length of the dimension and -1 means the dimension is dynamic.
- optional(optional): Whether the input or output is optional. By default, it is false.
- force_cpu(optional): Whether to force the input or output to be on CPU. By default, it is false.

### Model Definition

The model definition is derived from the Triton model configuration and extended to fit all the other backend requirements:

- name: Specifies the model's name. The model files are expected to reside in the model repository under a directory matching this name.
- backend: Defines the backend implementation of the model. The definition is hierarchical and the backend type can be specified at multiple levels. e.g. `backend: deepstream/nvinfer` means the backend uses "nvinfer" from "deepstream"; and `backend: triton/python/tensorrtllm` means the backend uses Triton with python and tensorrtllm plugin. Supported backends include:
  - deepstream: Deepstream backend, supporting "nvinfer" and "nvinferserver"
  - triton: Triton backend, supporting "python" and all the other triton backends
  - tensorrtllm/pytorch: TensorRT LLM backend with pytorch flow
  - polygraphy: Tensorrt backend through polygraphy
  - pytorch: Pytorch backend for models from Huggingface Transformers
  - dummy: Dummy backend for dry-run test without a model
- max_batch_size: The maximum batch size of the model.
- input: The input definition of the model.
- output: The output definition of the model.
- parameters (optional): The parameters of the model. This part is a custom section and is backend dependent.
- preprocessors (optional): list of the preprocessors used by the model.
- postprocessors (optional): list of the postprocessors used by the model.

When triton is used as the backend, all the [standard triton model parameters](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) are supported.


### Custom Preprocessors and Postprocessors

Custom preprocessors and postprocessors integrate user-defined Python code into the inference flow, offering a new paradigm for programming with neural network models.

Both custom preprocessors and postprocessors are defined using the following specification:
- kind: The kind of the processor: "custom" or "auto"
- name: The name of the processor. It must match the "name" defined in user implemented processor class.
- input: Specifies the names of the processor’s inputs in order. This defines how tensors from the inference flow are passed to the processor.
- output: Specifies the names of the processor’s output in order. This defines how the inference flow extracts the tensors from the processor.
- config: Defines the processor’s configuration as a dictionary. The contents are implementation-specific.

#### Custom Preprocessor/Postprocessor Implementation Requirements

When implementing a custom preprocessor or postprocessor, the class must adhere to the following specification:
- A class variable named "name" must be defined to uniquely identify the processor.
- An __init__ method must be implemented to accept a single argument: a dictionary containing the processor’s configuration (config).
- A __call__ method must be defined to accept multiple positional arguments. The number and order of these arguments must match the input definition in the configuration file.
- The __call__ method must return a tuple of tensors, in the order specified in the configuration file's output section.
- Input and output data are expected to be either NumPy arrays or PyTorch tensors, unless a custom data type is explicitly specified.

### Server Definition

The server definition is required at the top level unless the server type is set to "serverless". It specifies the server implementation along with the corresponding request and response templates. Both templates must be written in Jinja2, and are used to:
- Extract the necessary inputs from the server request.
- Format the inference outputs into the desired server response format.

The "responders" section allows users to map server implementations to operations defined in the OpenAPI specification. Each responder corresponds to a specific endpoint or operation.

Below are the supported responder types:

- infer: Performs the inference via the available models.
- add_file: upload a new file to the server as an asset.
- del_file: delete a file from the server.
- list_files: list all the files in the server.
- add_live_stream: add a live stream as an asset
- del_live_stream: delete a live stream from the asset pool
- list_live_stremas: list all the live streams known by the server

### Routing

The routes section is optional and typically used for more complex inference flows. It defines custom routing rules that control how data flows between different models in the pipeline.

The routes definition is a map where each entry specifies a connection between an input and an output:
- The key represents the source (input), and the value represents the destination (output).
- Both the input and output are defined using the format: <model_name>:<input_or_output_list>
- If the model name is omitted (e.g., :input_name), the input or output is assumed to be at the top level of the inference flow.

Here is an example of route definition:

```yaml
routes:
  ':["images"]': "visionenc",
  'visionenc:["features"]': 'vila1.5-13b:',
  'vila1.5-13b:["text"]': ':["summary"]'
```

This defines the routing logic for a multi-stage inference pipeline. The meaning of each route is as follows:

- The top-level input named "images" is routed to the "visionenc" model and the tensors named "image" will be passed to "visionenc"/
- The output named "features" from the "visionenc" model is routed to the input of the "vila1.5-13b" model. Since no input name is specified for "vila1.5-13b", tensors with name "features" will be passed to it without renaming.
- The output named "text" from the "vila1.5-13b" model is routed to the top-level output named "summary", which means tensors named "text" will be passed to top level after being renamed to "summary".

## Contributing

Contributions are welcome! Please feel free to submit a PR.


## Project status and roadmap

The project is under active development and the following features are expected to be supported in the near future:

- Support for more backends and frameworks such as VLLM and onnx runtime.
- Support for more model types such as speech models.
