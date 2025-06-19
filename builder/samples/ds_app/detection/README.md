This sample demonstrates how to build a deepstream application with Inference Builder using object detection models:
1. resnet
2. rtdetr

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "resnet", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/resnet' for the model to be correctly loaded.
Download the model files from the NGC catalog and put them in the $MODEL_REPO/resnet/ directory

To copy the other required model files to the $MODEL_REPO/resnet/ directory, run the following command:

```bash
cp builder/samples/ds_app/detection/resnet/* $MODEL_REPO/resnet/
```

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/samples/directory
```

## Generate the application package and build it into a docker image:

### Build the deepstream application package

```bash
python builder/main.py builder/samples/ds_app/detection/ds_detect.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

## Run the deepstream app with different inputs:

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

### Run with source config

```bash
docker run --rm --net=host --gpus all \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --source-config /opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/pipeline_api/deepstream_test5_app/source_list_dynamic.yaml
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