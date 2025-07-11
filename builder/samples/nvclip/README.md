# Introduction

This sample demonstrate how to create the inference package for an embedding NIM with tensorrt backend (polygrapy) and fastapi server.

# Prerequisites

The model used in this sample can be found from NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/nvclip_vit. Make sure you have the proper access right to the models, and download the checkpoints.

```bash
ngc registry model download-version "nvidian/tao-nvaie/multi_modal_foundation_models:NVCLIP_224_700M_ViTH14"
```

Prior to using the model for accelerated inference, we need to convert the checkpoints into onnx file and generate TensorRT engines accordingly.

First build the container image for TensorRT optimizer:

```bash
docker build -t trt-optimizer builder/samples/nvclip/optimizer
```

Then generate the engine files using the trt-optimizer

```bash
docker run -it --rm --gpus all \
           -v ~/.cache/model-repo/:/workspace/checkpoints/optimized \
           -v {Your download directory}/multi_modal_foundation_models_vNVCLIP_224_700M_ViTH14:/workspace/checkpoints/baseline \
           -e CHECKPOINT_NAME=nvclip_clipa_vit_h14_700M.ckpt \
           trt-optimizer
```

If the above process is correct, there'll be 2 folders appearing under model-repo directory:
- nvclip_clipa_vit_h14_700M_vision: for vision encoder
- nvclip_clipa_vit_h14_700M_text: for text encoder

Then copy the model config from source tree to the model folder:

```
cp builder/samples/nvclip/optimizer/configs/* ~/.cache/model-repo/nvclip_clipa_vit_h14_700M_text

# Build the NIM inference flow

```bash
python builder/main.py builder/samples/nvclip/tensorrt_nvclip.yaml -a builder/samples/nvclip/openapi.yaml -c builder/samples/nvclip/processors.py -o builder/samples/nvclip --server-type fastapi -t
```

# Build and run the docker image

```bash
cd builder/samples
docker compose up --build ms-nvclip
```

# Test the NIM with a client:

```bash
cd builder/samples/nvclip
./test_client.sh <sample.png>
```