This sample demonstrates how to build a deepstream application with Inference Builder using classification models:
1. changenet-classify
2. pcbclassification

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "pcbclassifier", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/pcbclassifier' for the model to be correctly loaded.
Download the model files from the NGC catalog and put them in the $MODEL_REPO/pcbclassifier/ directory

To copy the other required model files to the $MODEL_REPO/pcbclassifier/ directory, run the following command:

```bash
cp builder/samples/ds_app/classification/pcbclassification/* $MODEL_REPO/pcbclassifier/
```

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/samples/directory
```

## Generate the application package and build it into a docker image:

### Build the deepstream application package

```bash
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

## Run the deepstream app with image inputs:

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

**Note:**For changenet classification model, it requires two images as input. Here we are using the sample images provided in the sample_input directory as a reference.

```bash
docker run --rm --net=host --gpus all \
    -v $SAMPLE_INPUT:/sample_input \
    -v $MODEL_REPO:/workspace/models \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    deepstream-app \
    --media-url /sample_input/test_1.jpg /sample_input/golden_1.jpg \
    --mime image/jpeg image/jpeg
```