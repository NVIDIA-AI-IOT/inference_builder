## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using segmentation models:
1. citysemsegformer: deployable_onnx_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer

## Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference-builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference-builder root directory.

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
mkdir ~/.cache/model-repo/
sudo chmod -R 777 ~/.cache/model-repo/
export MODEL_REPO=~/.cache/model-repo
```

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/{model-name}/ directory, then copy the other required configurations to the same directory:

```bash
ngc registry model download-version "nvidia/tao/citysemsegformer:deployable_onnx_v1.0"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv citysemsegformer_vdeployable_onnx_v1.0 $MODEL_REPO/citysemsegformer
chmod 777 $MODEL_REPO/citysemsegformer
cp -r builder/samples/ds_app/segmentation/citysemsegformer/* $MODEL_REPO/citysemsegformer/
```

## Generate the deepstream application package and build it into a container image:

Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment and be sure you're in the inference-builder folder, then activate your virtual environment:

```bash
source .venv/bin/activate
```

You need to export your GITLAB_TOKEN for pulling source dependencies from gitlab

```bash
export GITLAB_TOKEN={Your GitLab Token}
```

### For x86 Architecture

```bash
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

**Note:** The TensorRT engine is generated during the first time run and it takes several minutes.

**Note:** You can optionally set the `$SAMPLE_INPUT` environment variable to point to your samples directory if you perform inference on media files in your host.

**Note:** By default the inference results are printed to the console, you can save it to a json file if appending `-s result.json` to the commands

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

**Note:** When you set `enable_display: true` under the `render_config` section of [ds_segformer.yaml](ds_segformer.yaml), you need to have a display on your host and run both commands in this order to give the container access to it. For more information about render configuration options, see the [render configuration section](../README.md#render-configuration).

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
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
docker run --rm --net=host --gpus all --runtime=nvidia \
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
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# Replace rtsp://<url_path> with your actual RTSP stream URL
docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url rtsp://<url_path> \
    --mime video/mp4
```

### Run with image input

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").

# Note /sample_input/test_1.jpg is just a placeholder for any image present in $SAMPLE_INPUT directory

docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test_1.jpg \
    --mime image/jpeg
```