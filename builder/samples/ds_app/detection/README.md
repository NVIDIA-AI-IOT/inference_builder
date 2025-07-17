## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using object detection models:
1. resnet
2. rtdetr

## Prerequisites

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "rtdetr", you must put all the model files including nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/rtdetr' for the model to be correctly loaded.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/rtdetr/ directory, then copy the other required model files to the same directory:

```bash
cp builder/samples/ds_app/detection/rtdetr/* $MODEL_REPO/rtdetr/
```

## Generate the deepstream application package and build it into a container image

```bash
python builder/main.py builder/samples/ds_app/detection/ds_detect.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

## Run the deepstream app with different inputs:

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/samples/directory
```

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

### Run with source config

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --source-config /opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/pipeline_api/deepstream_test5_app/source_list_dynamic.yaml
```

**Note:** To use a custom source configuration file, you need to mount your file into the docker container and reference it from within the container's filesystem. This allows you to use your own source configuration instead of the default one.

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v $SAMPLE_INPUT:/workspace/inputs \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --source-config /workspace/inputs/source_list_dynamic.yaml
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

## MV3DT (Multi-View 3D Tracking)

This section demonstrates how to set up and run the MV3DT model for multi-view 3D tracking.

### Model Setup

1. **Download Model Files:**
   Download the model content from [Google Drive](https://drive.google.com/drive/folders/1EiLjPYjGeIF2duElHlH11lu94_fO5WEZ?usp=drive_link) into your `$MODEL_REPO` directory.

2. **Copy Required Files:**
   ```bash
   cp builder/samples/ds_app/detection/PeopleNetTransformer/* $MODEL_REPO/PeopleNetTransformer/
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

### Build and Run

**Generate the deepstream application package and build it into a container image:**

```bash
python builder/main.py builder/samples/ds_app/detection/ds_mv3dt.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

**Run with multi-camera video input:**

```bash
docker run --rm --net=host --gpus all \
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

