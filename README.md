# Inference Builder

## Overview

Inference Builder is a tool that allows developers to generate inference code from various code templates for different VLM/CV NIMs. It takes a comprehensive inference configuration file and the corresponding OpenAPI spec. For some NIMs, custom code snippet would also be needed.

There are three main components in the Inference Builder:

- Inference code templates: The code templates provides reusable code snippets for different inference backends and frameworks, including Triton Inference Server, TensorRT, Deepstream and so on. These templates are optimized and tested, so to be used by any models with instantiated input/output and configuration.
- Python library for common inference utilities: This library provides common utilities for inference, including data loading, media processing, and configuration management.
- Command line tool for generating inference code: This application takes the inference configuration file and the OpenAPI spec as input, and generates the inference code for the corresponding NIM by piecing together the code templates and the common utilities.

## Getting started

### Clone the repository

```bash
git clone https://gitlab-master.nvidia.com/chunlinl/nim-templates.git
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
$ cd nim-templates
$ pip3 install -r requirements.txt

```

## NIM Dependencies

Ensure nvidia runtime added to `/etc/docker/daemon.json`

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

Install Nim Tools

```bash
$ pip install nimtools --index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

#Check installation
$ nim_builder --version
```

## Usage
The project provides developers an easy-to-use command line tool to generate inference codes for various VLM/CV NIMs. Before running the tool, developers need to prepare a comprehensive configuration file to define the inference flow of the NIM, in addition, there needs an OpenAPI spec to define the NIM endpoints and parameters. For some NIMs, custom code snippet would also be needed.

For generating the inference code with the corresponding server implementation, developers can run the following command:

```bash
pyton builder/main.py
usage: Inference Builder [-h] [--server-type [{triton}]] [-o [OUTPUT_DIR]] [-a [API_SPEC]] [-c [CUSTOM_MODULE ...]] [-x] [-t] config
```

There're several builtin samples under _samples_ folder for generating various CV NIMs and VLM NIMs.

## Configuration File

A configuration file is a YAML file that defines the inference flow of the NIM. It contains the following sections:

- name: The name of the NIM. It will be used as the name of the folder to save the generated inference code (or the name of the tarball if '-t' is specified).
- models: List of model definitions. A inference flow can contain multiple models, each might be implemented by different backends to achieve better performance.
- input: Defines the top level input parameters of the NIM, and each input is extracted from the server request.
- output: Defines the top level output parameters of the NIM, and each output is wrapped into the server response.
- server: Defines the server endpoints and the corresponding request and response types.

A configuration file can be as simple as the following example:

```yaml

name: "segformer"

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
  endpoints:
    infer:
      path: /v1/inference
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
    data_type: TYPE_UINT8
    dims: [544, 960, 3]
  - name: format
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: false
  output:
  - name: output
    data_type: TYPE_CUSTOM_DS_METADATA
    dims: [ -1 ]
  parameters:
    infer_config_path: /opt/nim/.cache/model-repo/segformer/config_nvinfer.yaml
```

With the above configuration file, it is sufficient to generate the inference code for the segformer NIM with Deepstream backend which takes images as input and generates segmentation masks as output.

### Model Definition

The model definition is derived from the Triton model configration and extended to fit all the other backend requirements:

- name: The name of the model.
- backend: The backend implementation of the model. The definition is hierarchical and the backend type can be specified at multiple levels. e.g. `backend: deepstream/nvinfer` means the model will be implemented by Deepstream with nvinfer backend; and `backend: triton/python/tensorrtllm` means the model will be implemented by Triton with python backend and tensorrtllm plugin.
- max_batch_size: The maximum batch size of the model.
- input: The input parameters of the model.
- output: The output parameters of the model.
- parameters (optional): The parameters of the model. This part is a custom section and is backend dependent.
- tokenizer (optional): The tokenizer used by the model:
  - type: The type of the tokenizer.
  - encoder: defines the input and output names of the encoder
  - decoder: defines the input and output names of the decoder
- preprocessors (optional): list of the preprocessors used by the model.
  - kind: The kind of the preprocessor. A 'custom' preprocessor is a user defined preprocessor and the code snippet conforming to the subsequent definition is required.
  - name: The name of the preprocessor.
  - input: The input names of the preprocessor.
  - output: The output names of the preprocessor.
  - config: The configuration map of the preprocessor. This part is a custom section and is implementation dependent.
- tensorrt_engine (optional): The path to the tensorrt engine file. Only required if backend is tensorrt.

When triton is used as the backend, all the standard triton model parameters are supported.


### Input and Output Definition

Input and Output definitions are required both in top level and model level.Each input and output definition contains the following sections:

- name: The name of the input or output.
- data_type: The data type of the input or output. In addition to the basic data types defined by Triton, the project also supports following custom data types:
  - TYPE_CUSTOM_IMAGE_BASE64: The input or output is a base64 encoded image and will be decoded to a image tensor.
  - TYPE_CUSTOM_BINARY_BASE64: The input or output is a base64 encoded string and will be decoded to a binary tensor.
- dims: The dimensions of the input or output in the form of a list. Each item in the list specifies the maximum length of the dimension and -1 means it is dynamic.
- optional: Whether the input or output is optional.
- force_cpu: Whether to force the input or output to be on CPU.

### Server Definition

The server definition is required in top level. It defines the server inference endpoint and the corresponding request and response templates for it. Both request and response templates are Jinja2 templates and are used to extract required input and output data from the server request and response.

### Routing

The routing section is optional and only used in rather complex inference flows. It defines the routing rules for the inference flow when multiple models are involved and multiple data flows are possible. Definition of 'routes' is a map and the key is the input and the value is the output. Both input and output are defined with the name of the model and the list of its tensors.


## Contributing

Contributions are welcome! Please feel free to submit a PR.


## Project status and roadmap

The project is under active development and the following features are expected to be supported in the near future:

- Support for video and streaming.
- Support for more backend frameworks.
- Support for more server implementations.
