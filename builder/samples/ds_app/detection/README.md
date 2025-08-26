## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using object detection models:
1. rtdetr: trafficcamnet_transformer_lite_vdeployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet_transformer_lite

## Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference-builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference-builder root directory.

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
mkdir ~/.cache/model-repo/
sudo chmod -R 777 ~/.cache/model-repo/
export MODEL_REPO=~/.cache/model-repo
```

For example: if you define a model with name "rtdetr", you must put all the model files including nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/rtdetr' for the model to be correctly loaded.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/rtdetr/ directory, then copy the other required configurations to the same directory:

```bash
ngc registry model download-version "nvidia/tao/trafficcamnet_transformer_lite:deployable_v1.0"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv trafficcamnet_transformer_lite_vdeployable_v1.0 $MODEL_REPO/rtdetr
chmod 777 $MODEL_REPO/rtdetr
cp -r builder/samples/ds_app/detection/rtdetr/* $MODEL_REPO/rtdetr/
```

## Generate the DeepStream Application Package and Build Container Image

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
python builder/main.py builder/samples/ds_app/detection/ds_detect.yaml \
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
python builder/main.py builder/samples/ds_app/detection/ds_detect.yaml \
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

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

**Note:** When you set `enable_display: true` under the `render_config` section of your inference builder config, you need to have a display on your host and run both commands in this order to give the container access to it. For more information about render configuration options, see the [render configuration documentation](https://gitlab-master.nvidia.com/DeepStreamSDK/inference-builder/-/tree/main/builder/samples/ds_app?ref_type=heads#render-configuration).


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

### Run with source config

```bash
# source-config: path to the source configuration file that defines input sources
docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --source-config /opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/pipeline_api/deepstream_test5_app/source_list_dynamic.yaml
```

**Note:** To use a custom source configuration file, you need to mount your file into the docker container and reference it from within the container's filesystem. This allows you to use your own source configuration instead of the default one.

```bash
# source-config: path to the source configuration file that defines input sources
# /workspace/inputs/source_list_dynamic.yaml is just a placeholder for any config present in $SAMPLE_INPUT directory
docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v $SAMPLE_INPUT:/workspace/inputs \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --source-config /workspace/inputs/source_list_dynamic.yaml
```

### Run with image input

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# /sample_input/test_1.jpg is just a placeholder for any image present in $SAMPLE_INPUT directory
docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test_1.jpg \
    --mime image/jpeg
```

## MV3DT (Multi-View 3D Tracking)

This section demonstrates how to set up and run the MV3DT specific models for multi-view 3D tracking.

### Model Setup

1. **Download Model Files:**
   Download the model content from [Google Drive](https://drive.google.com/drive/folders/1EiLjPYjGeIF2duElHlH11lu94_fO5WEZ?usp=drive_link) into your `$MODEL_REPO` directory.

    **Note:** MV3DT-specific model files will be made publicly available in a future release (required for DS-8.0). The download instructions above will be updated once the models are hosted on the official NGC catalog or other public repositories.
    
    **Note:** After downloading the models inside $MODEL_REPO, you would have four model directories like this:
    - PeopleNetTransformer
    - BodyPose3DNet
    - ReID-MTMC
    - PeopleNet2.6.3

2. **Copy Required Files:**
   ```bash
   cp -r builder/samples/ds_app/detection/PeopleNetTransformer/* $MODEL_REPO/PeopleNetTransformer/
   ```

3. **Create Output Directories:**
   ```bash
   mkdir -p $MODEL_REPO/PeopleNetTransformer/infer-kitti-dump
   mkdir -p $MODEL_REPO/PeopleNetTransformer/tracker-kitti-dump
   mkdir -p $MODEL_REPO/PeopleNetTransformer/trajDumps
   ```

4. **Clean up existing ModelEngine files (optional):**
   ```bash
   # Remove existing TensorRT engine files to force regeneration for your system
   # This is recommended when switching between different GPU architectures or TensorRT versions
   rm -f $MODEL_REPO/PeopleNetTransformer/*.engine
   rm -f $MODEL_REPO/BodyPose3DNet/*.engine
   rm -f $MODEL_REPO/ReID-MTMC/*.engine
   rm -f $MODEL_REPO/PeopleNet2.6.3/*.engine
   ```

### Sample Data Setup

1. **Download Sample Streams:**
   Download the sample content from [Google Drive](https://drive.google.com/drive/folders/1elBteIllmbdDSE0EMEiYjG_ZTwKrKXwE?usp=drive_link) into your `$SAMPLE_INPUT` directory.

2. **Set Environment Variable:**
   ```bash
   export SAMPLE_INPUT=/path/to/MTMC_Warehouse_Synthetic_4cam/
   ```

### MQTT Setup for Tracker

Install the MQTT broker and clients required for the tracker:

```bash
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
sudo apt update
sudo apt install mosquitto mosquitto-clients
```

Launch mosquitto broker

```bash
mosquitto -p 1883
```

### Build and Run

**Generate the deepstream application package and build it into a container image:**

### For x86 Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/detection/ds_mv3dt.yaml \
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
python builder/main.py builder/samples/ds_app/detection/ds_mv3dt.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

**Run with multi-camera video input:**

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
docker run --rm --net=host --gpus all --runtime=nvidia \
    -v $MODEL_REPO:/workspace/models \
    -v $SAMPLE_INPUT:/workspace/inputs \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /workspace/inputs/videos/Warehouse_Synthetic_Cam001.mp4 \
                /workspace/inputs/videos/Warehouse_Synthetic_Cam002.mp4 \
                /workspace/inputs/videos/Warehouse_Synthetic_Cam003.mp4 \
                /workspace/inputs/videos/Warehouse_Synthetic_Cam004.mp4 \
    --mime video/mp4 video/mp4 video/mp4 video/mp4
```

### Results

Once the run is complete, the following output data will be populated in the respective directories:

- **`infer-kitti-dump`** - Inference kitti data
- **`tracker-kitti-dump`** - Tracker Kitti data
- **`trajDumps`** - Trajectory dump data

