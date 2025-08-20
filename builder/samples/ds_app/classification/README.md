## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using classification models:
1. changenet-classify: visual_changenet_nvpcb_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification
2. pcbclassification: deployable_v1.1 from https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification

## Prerequisites

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "pcbclassification", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/pcbclassification' for the model to be correctly loaded.

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/pcbclassification/ directory, then copy the other required model files to the same directory:

```bash
ngc registry model download-version "nvaie/pcbclassification:deployable_v1.1"
mv pcbclassification_vdeployable_v1.1 $MODEL_REPO/pcbclassification
chmod 777 $MODEL_REPO/pcbclassification
cp builder/samples/ds_app/classification/pcbclassification/* $MODEL_REPO/pcbclassification/
```

The sample steps apply to the other classification model: changenet-classify


## Generate the deepstream application package and build it into a container image

**Note:** For Thor and Spark, please use "-f builder/samples/ds_app/Dockerfile.tegra"

Set up your Gitlab token:

```bash
export GITLAB_TOKEN={Your Gitlab Token}
```

For pcbclassification sample, please use ds_pcb.yaml as the configuration:

```bash
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

For changenet-classify sample, please use ds_changenet.yaml as the configuration:

```bash
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```


## Run the deepstream app with image inputs:

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/samples/directory
```

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

**Note:** For changenet classification model, it requires two images as input. Here we are using the sample images provided in the sample_input directory as a reference.

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

For classification samples, the output is a list of labels defined in labels.txt. e.g, for pcbclassification model, the output label will be "missing" if there is a part missing from the input image.