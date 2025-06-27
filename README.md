# Inference Builder

## Overview

Inference Builder is a tool for generating inference pipelines and integrating them into a microservice or standalone application automatically. It takes an inference configuration file, an OpenAPI specification while integrated with an HTTP server as inputs, and in some cases, also requires custom code snippets.

The output of the tool is a Python package that can be used to build a microservice container image with a customized Dockerfile.

The Inference Builder consists of three major components:

- Code templates: These are reusable modules for various inference backends and frameworks, as well as for API servers. They are optimized and tested, making them suitable for any model with specified inputs, outputs, and configuration parameters.
- Common inference flow: This is the backbone code that standardizes the inference flow covering data loading/pre-processing, model inference, and post-processing, and its integration with the API server. Different inference backends and frameworks can be plugged in to this flow to achieve optimal performance.
- Command line tool: It generates the source code package by combining the code templates and the common inference flow. In addition, it automatically generates corresponding test cases and evaluation scripts.

## Getting started

### Clone the repository

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/DeepStreamSDK/inference-builder.git
git submodule update --init
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
The project provides developers an easy-to-use command line tool to generate inference and server implementation. Before using the tool, developers need to prepare a comprehensive configuration file to define the inference flow, in addition, there needs an OpenAPI spec to define server endpoints and parameters.

For generating the inference code with the corresponding server implementation, developers can run the following command:

```bash
python builder/main.py
usage: Inference Builder [-h] [--server-type [{triton}]] [-o [OUTPUT_DIR]] [-a [API_SPEC]] [-c [CUSTOM_MODULE ...]] [-x] [-t] config
```

There're several builtin samples under _samples_ folder for generating inference code and server implementation with different inference backends and frameworks.

## Configuration File

A configuration file is a YAML file that defines the inference flow and server implementation. It contains the following sections:

- name: The name of the microservice. It will be used as the name of the folder to save the generated inference code (or the name of the tarball if '-t' is specified), and it will also be used to identify the microservice during runtimme.
- model_repo: The path to the model repository where all the required model files are stored.
- models: List of model definitions. A inference flow can contain multiple models, each might be implemented with different backends to achieve optimal performance.
- input: Defines the top-level input parameters of the inference flow and each input can be extracted from the server request.
- output: Defines the top-level output parameters of the inference flow and each output could be wrapped into the server response.
- server: Defines the server implemenation and the corresponding request and response mappings.
- routes(optional): Defines the routing rules for the inference flow when multiple models are involved, and multiple data flows are possible.
- postprocessors(optional): Defines the top-level postprocessors for the inference flow, which processes the output data before sending them to the server.

A configuration file can be as simple as the below example:

```yaml

name: "segformer"
model_repo: "/opt/model-repo"

input:
- name: images
  data_type: TYPE_CUSTOM_BINARY_BASE64
  dims: [ -1 ]
  optional: false

- name: format
  data_type: TYPE_STRING
  dims: [ -1 ]
  optional: false

output:
  - name: output
    data_type: TYPE_CUSTOM_DS_METADATA
    dims: [ -1 ]

server:
  responders:
    infer:
      operation: _infer_inference_post
      requests:
        InferenceRequest: >
          {
            {% set image_items = request.input if request.input is iterable else [request.input] %}
            "images": [
              {% for item in image_items %}
                {{ item|replace('data:image\/[a-zA-Z0-9.+-]+;base64,', '')|tojson }}{% if not loop.last %}, {% endif %}
              {% endfor %}
            ],
            "format": [
              {% for item in image_items %}
                {{ item|extract('data:image\/(\w+);base64,')|tojson }}{% if not loop.last %}, {% endif %}
              {% endfor %}
            ]
          }
      responses:
        InferenceResponse: >
          {
            "data": [
              {% for item in response.output %} {
                "mask": {{item.data.seg_map}}
              } {% if not loop.last %}, {% endif %} {% endfor %} ],
          }

models:
- name: segformer
  backend: deepstream/nvinfer
  max_batch_size: 1
  input:
  - name: images
    data_type: TYPE_CUSTOM_DS_IMAGE
    dims: [-1, -1, 3]
  - name: format
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: false
  output:
  - name: output
    data_type: TYPE_CUSTOM_DS_METADATA
    dims: [ -1 ]
  parameters:
    infer_config_path:
      - config_nvinfer.yaml
```

With the above configuration file, it is sufficient to generate the inference code for the segformer model with Deepstream backend taking images as input and generating segmentation masks as output.

### Model Definition

The model definition is derived from the Triton model configuration and extended to fit all the other backend requirements:

- name: The name of the model. With the name specified, the model files will be stored in the model repository with the name as the folder name.
- backend: The backend implementation of the model. The definition is hierarchical, and the backend type can be specified at multiple levels. e.g. `backend: deepstream/nvinfer` means the model will be implemented by Deepstream with nvinfer backend; and `backend: triton/python/tensorrtllm` means the model will be implemented by Triton with python backend and tensorrtllm plugin. Below are the supported backends:
  - deepstream: Deepstream backend, supporting nvinfer and nvinferserver
  - triton: Triton backend, supporting python and other triton backends
  - tensorrtllm: TensorRT LLM backend
  - polygraphy: Tensorrt backend through polygraphy
- max_batch_size: The maximum batch size of the model.
- input: The input definition of the model.
- output: The output definition of the model.
- parameters (optional): The parameters of the model. This part is a custom section and is backend dependent.
- preprocessors (optional): list of the preprocessors used by the model.
- postprocessors (optional): list of the postprocessors used by the model.

When triton is used as the backend, all the [standard triton model parameters](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) are supported.


### Custom Preprocessors and Postprocessors

Bother custom preprocessors and postpocessors are defined by below specification:
- kind: The kind of the processor: "custom" or "auto"
- name: The name of the processor.
- input: names of the processor's input in order.
- output: names of the processor's output in order.
- config: The configuration of the processor, which is a dictionary and implementation-specific. Two standard keys are supported:
  - model_home: The home directory of the model, which is used to locate the files used by the processor. Overridable by the user.
  - device_id: The device id of the tensor data.

  When user implements a custom preprocessor or postprocessor, the class must follow the following specification:
  - A class variable of 'name' must be uniquely defined to identify the processor.
  - A __init__ method must be defined to take a config dictionary as argument.
  - A __call__ method must be defined to take multiple positional arguments. The number of positional arguments must match the definition in the configuration file.
  - The __call__ method must return a tuple of tensors in the order of the definition in the configuration file.
  - The data passed to the __call__ method and the returned data are assumed to be a numpy arrary or torch tensor unless custom data type is specified.

### Input and Output Definition

Input and Output definitions are required both at the top level and model level. And each input and output definition contains the following sections:

- name: The name of the input or output. Tensors that passed through the input or output are named after it.
- data_type: The data type of the input or output. In addition to the [basic data types defined by Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes), the project also supports following custom data types:
  - TYPE_CUSTOM_IMAGE_BASE64: For input, the data will be decoded to a image tensor, and for output, the image tensor will be encoded to a base64 string.
  - TYPE_CUSTOM_BINARY_BASE64: For input, the data will be decoded to a byte tensor, and for output, the byte tensor will be encoded to a base64 string.
  - TYPE_CUSTOM_IMAGE_ASSETS: The data contains strings that identify the uploaded image assets, which are automatically passed to image decoder and converted to image tensors
  - TYPE_CUSTOM_VIDEO_ASSETS: The data contains strings that identify the uploaded video assets, which are automatically passed to video decoder and converted to image tensors frame by frame.
  - TYPE_BINARY_URLS: For input, the data will be converted to a list and treated as urls, and for output, the list of urls will be converted to a tensor.
  - TYPE_CUSTOM_DS_IMAGE: Encoded image specificially for Deepstream pipeline
  - TYPE_CUSTOM_DS_MIME: Mime type used by Deepstream pipeline to determine the decoder type
  - TYPE_CUSTOM_DS_METADATA: Structured output data specifically for Deepstream pipeline
- dims: The dimensions of the input or output in the form of a list. Each item in the list specifies the maximum length of the dimension and -1 means it is dynamic.
- optional(optional): Whether the input or output is optional. By default, it is false.
- force_cpu(optional): Whether to force the input or output to be on CPU. By default, it is false.

### Server Definition

The server definition is required at the top level. It defines the server implementation and the corresponding request and response templates for it. Both request and response templates are Jinja2 templates and are used to extract required input from server request or format output to server response.

Under responsers, user can map the server implementation to the operation defined in the OpenAPI specification. Below are the supported responder types:

- infer: The inference operation.
- add_file: upload a new file to the server.
- del_file: delete a file from the server.
- list_files: list all the files in the server.

### Routing

The routes section is optional and only used in rather complex inference flows. It defines the routing rules for the inference flow.

Definition of 'routes' is a map, and the key is the input, and the value is the output. Both input and output are defined with the name of the model and the list of its I/Os delimited by ':'. If the model name is omitted, it means the input or output is in the top level.

For example, the following route definition:

```yaml
routes:
  ':["images"]': "visionenc",
  'visionenc:["features"]': 'vila1.5-13b:',
  'vila1.5-13b:["text"]': ':["text"]'
```

means:

- The top level input named 'images' will be connected to the input named 'images' of 'visionenc' model.
- The output named 'features' of 'visionenc' model will be connected to the input named 'images' of 'vila1.5-13b' model.
- The output named 'text' of 'vila1.5-13b' model will be connected to the top level output named 'text'.

## Contributing

Contributions are welcome! Please feel free to submit a PR.


## Project status and roadmap

The project is under active development and the following features are expected to be supported in the near future:

- Support for more backends and frameworks such as VLLM and onnx runtime.
- Support for more model types such as speech models.
