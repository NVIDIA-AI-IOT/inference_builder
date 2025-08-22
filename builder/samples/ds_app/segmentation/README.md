## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using segmentation models:
1. citysemsegformer: deployable_onnx_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer
2. mask2former: mask2former_swint_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask2former

**Note:** For both citysemsegformer and mask2former models, the steps are the same as described below. Users just need to replace the model name from "citysemsegformer" to "mask2former" in the commands and directory names.

## Prerequisites

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "citysemsegformer", you must put all the model files including nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/citysemsegformer' for the model to be correctly loaded.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/citysemsegformer/ directory, then copy the other required configurations to the same directory:

```bash
ngc registry model download-version "nvidia/tao/citysemsegformer:deployable_onnx_v1.0"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv citysemsegformer_vdeployable_onnx_v1.0 $MODEL_REPO/citysemsegformer
chmod 777 $MODEL_REPO/citysemsegformer
cp builder/samples/ds_app/segmentation/citysemsegformer/* $MODEL_REPO/citysemsegformer/
```

## Generate the deepstream application package and build it into a container image:

### For x86 Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/segmentation/ds_segformer.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    builder/samples/ds_app
```

### For Tegra Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/segmentation/ds_segformer.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

## Run the deepstream app with different inputs:

**Note:** You can optionally set the `$SAMPLE_INPUT` environment variable to point to your samples directory if you perform inference on media files in your host.

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

**Note:** When you set `enable_display: true` under the `render_config` section of your inference builder config, you need to have a display on your host and run both commands in this order to give the container access to it.

First, set the display environment variable:
```bash
export DISPLAY=:0  # or :1 depending on your system
```

Then, allow X server connections from any host:
```bash
xhost +
```

If the configuration is successful, you will see this message in the log: `access control disabled, clients can connect from any host`.

### Run with video input

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
    --mime video/mp4
```

### Run with RTSP input

**Note:** Replace `rtsp://<url_path>` (which is just a placeholder) with your actual RTSP stream URL. The application supports various RTSP stream formats including H.264, H.265, and MJPEG.

```bash
# Replace rtsp://<url_path> with your actual RTSP stream URL
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url rtsp://<url_path> \
    --mime video/mp4
```

### Run with image input

```bash
# /sample_input/test_1.jpg is just a placeholder for any image present in $SAMPLE_INPUT directory
docker run --rm --net=host --gpus all \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test_1.jpg \
    --mime image/jpeg
```