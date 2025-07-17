# Introduction

This sample demonstrates how to build an inference pipeline for models that use triton backend. It also shows how to incorporate CVCUDA in customized processors to accelerate media processing.

The sample has been tested on following platforms
- RTX 4090
- H100
- A6000

# Prerequisites

The model used in the sample can be found on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_segmentation_landsatscd

## Download the model file using NGC CLI:

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_segmentation_landsatscd:deployable_v1.2"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv visual_changenet_segmentation_landsatscd_vdeployable_v1.2 ~/.cache/model-repo/visual_changenet
```

## Create the optimized TensorRT Engine:

```bash
docker run -it --rm --gpus all \
-v ~/.cache/model-repo/visual_changenet/:/changenet \
nvcr.io/nvidia/tensorrt-pb25h1:25.03.02-py3 \
trtexec --onnx=/changenet/changenet_768.onnx --saveEngine=/changenet/model.plan --fp16
```

# Build the NIM inference flow:

```bash
python builder/main.py \
  --server-type triton \
  -a builder/samples/changenet/openapi.yaml \
  -c builder/samples/changenet/processors.py \
  -o builder/samples/changenet -t \
  builder/samples/changenet/trt_changenet.yaml
```

# Build and run the docker image:

```bash
cd builder/samples
docker compose up --build ms-changenet
```

# Test the NIM with a client:

```bash
cd builder/samples/changenet
python nim_client.py --host 127.0.0.1 --port 8803 --file <test1.jpg> <test2.jpg>
```