## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using segmentation models:
1. citysemsegformer: deployable_onnx_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer
2. mask2former: mask2former_swint_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask2former

## Prerequisites

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "citysemsegformer", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/citysemsegformer' for the model to be correctly loaded. Same steps apply for mask2former.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/citysemsegformer/ directory, then copy the other required model files to the same directory:

```bash
ngc registry model download-version "nvidia/tao/citysemsegformer:deployable_onnx_v1.0"
mv citysemsegformer_vdeployable_onnx_v1.0 $MODEL_REPO/citysemsegformer
chmod 777 $MODEL_REPO/citysemsegformer
cp builder/samples/ds_app/segmentation/citysemsegformer/* $MODEL_REPO/citysemsegformer/
```

## Generate the deepstream application package and build it into a container image:

**Note:** For Tegra Thor and DGX Spark, please use "-f builder/samples/ds_app/Dockerfile.tegra"

```bash
export GITLAB_TOKEN={Your Gitlab Token}
python builder/main.py builder/samples/ds_app/segmentation/ds_segformer.yaml -o builder/samples/ds_app --server-type serverless -t \
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
    --mime video/mp4
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
    --media-url /sample_input/test_1.jpg \
    --mime image/jpeg
```