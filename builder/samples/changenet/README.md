# Introduction

This sample demonstrates how to build an inference pipeline for models that use triton backend. It also shows how to incorporate CVCUDA in customized processors to accelerate media processing.

We provide a sample Dockerfile for the example, which you can use to build a Docker image and test the microservice on any x86 system with an NVIDIA Ampere, Hopper, and Blackwell GPU.

# Prerequisites

The model used in the sample can be found on NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_segmentation_landsatscd

Before downloading the model file, you need to set up your model repository:

```bash
mkdir -p ~/.cache/model-repo && chmod 777 ~/.cache/model-repo
export MODEL_REPO=~/.cache/model-repo
```

## Download the model file using NGC CLI:

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_segmentation_landsatscd:deployable_v1.2"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv visual_changenet_segmentation_landsatscd_vdeployable_v1.2 $MODEL_REPO/visual_changenet
# Change the access right for docker command
chmod 777 $MODEL_REPO/visual_changenet
chmod 666 $MODEL_REPO/visual_changenet/* # correct the permissions in case they're wrong
```

Run `ls $MODEL_REPO/visual_changenet -l`, be sure you have `changenet_768.onnx` in your model folder and the file permissions are correct.

## Create the optimized TensorRT Engine:

```bash
docker run -it --rm --gpus all \
-v ~/.cache/model-repo/visual_changenet/:/changenet \
nvcr.io/nvidia/tensorrt-pb25h1:25.03.02-py3 \
trtexec --onnx=/changenet/changenet_768.onnx --saveEngine=/changenet/model.plan --fp16
```

Run `ls $MODEL_REPO/visual_changenet -l`, be sure you have `model.plan` generated in your model folder and the file permissions are correct. The process takes approximately 10 minutes and you only need to run it once unless the model file changes.

# Build the inference flow:

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment, and you're already in the inference-builder folder.

The following command activates the virtual environment and generates the inference source package in `builder/samples/changenet`

```bash
source .venv/bin/activate
python builder/main.py \
  --server-type triton \
  -a builder/samples/changenet/openapi.yaml \
  -c builder/samples/changenet/processors.py \
  -o builder/samples/changenet -t \
  builder/samples/changenet/trt_changenet.yaml
```

# Build and run the docker image:

The process takes about 10 minutes, though the actual time may vary depending on your hardware and network bandwidth.

```bash
cd builder/samples
docker compose up --build ms-changenet
```

# Test the microservice with a client

After the server has started successfully, open a new terminal in the inference-builder folder and launch the client to compare two sample images. A display environment is required for the client to visualize the difference between the two input pictures.

To avoid errors related to display, you need to set the DISPLAY environment variable based on your system:

```bash
export DISPLAY=:0  # or :1 depending on your system
```

Then, allow X server connections from any host:
```bash
xhost +
```

If the configuration is successful, you will see this message in the log: `access control disabled, clients can connect from any host`.

You need to export your GITLAB_TOKEN for pulling source dependencies from gitlab

```bash
export GITLAB_TOKEN={Your GitLab Token}
```

Then start the client:

```bash
source .venv/bin/activate && cd builder/samples/changenet
python nim_client.py --host 127.0.0.1 --port 8803 --file test_0.jpg golden_0.jpg
```

A new window shows the difference between two images.