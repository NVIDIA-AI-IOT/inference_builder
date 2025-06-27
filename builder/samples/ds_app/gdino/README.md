This sample demonstrates how to build a deepstream application with Inference Builder using grounding dino  models:
1. gdino
2. mask_gdino

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "gdino", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/gdino' for the model to be correctly loaded.
Download the model files from the NGC catalog and put them in the $MODEL_REPO/gdino/ directory

To copy the other required model files to the $MODEL_REPO/gdino/ directory, run the following command:

```bash
cp builder/samples/ds_app/gdino/gdino/* $MODEL_REPO/gdino/
```

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/samples/directory
```

## Generate the application package and build it into a docker image:

### Build the deepstream application package

```bash
python builder/main.py builder/samples/ds_app/gdino/ds_gdino.yaml -o builder/samples/ds_app --server-type serverless -c builder/samples/tao/processors.py -t \
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
    --text "car"
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
    --mime image/jpeg
    --text "car"
```