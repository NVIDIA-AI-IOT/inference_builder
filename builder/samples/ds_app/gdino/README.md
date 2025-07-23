## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using grounding dino  models:
1. gdino: grounding_dino_swin_tiny_commercial_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino
2. mask_gdino: mask_grounding_dino_swin_tiny_commercial_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino

## Prerequisites

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "gdino", you must put all the model files include nvconfig, preprocess config, onnx, etc. to a single directory and map it to '/workspace/models/gdino' for the model to be correctly loaded.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/gdino/ directory, then copy the other required model files to the same directory:

```bash
cp builder/samples/ds_app/gdino/gdino/* $MODEL_REPO/gdino/
```

## Generate the deepstream application package and build it into a container image:

```bash
python builder/main.py builder/samples/ds_app/gdino/ds_gdino.yaml -o builder/samples/ds_app --server-type serverless -c builder/samples/tao/processors.py -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

## Run the deepstream app with different inputs:

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory if you perform inference on media files in you host.

**Note:** You need to have a display on your host and run `xhost +` to give the container access to it if you set enable_display to true in your render_config.

### Run with video input

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 \
    --mime video/mp4 \
    --text "car"
```

### Run with RTSP input

**Note:** Replace `rtsp://<url_path>` with your actual RTSP stream URL. The application supports various RTSP stream formats including H.264, H.265, and MJPEG.

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url rtsp://<url_path> \
    --mime video/mp4
```

**Examples:**

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url rtsp://127.0.0.1/video1 \
    --mime video/mp4
```

### Run with image input


```bash
docker run --rm --net=host --gpus all \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test.jpg \
    --mime image/jpeg \
    --text "car"
```