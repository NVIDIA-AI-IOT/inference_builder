## Introduction

This sample demonstrates how to build a deepstream application with Inference Builder using classification models:
1. changenet-classify: visual_changenet_nvpcb_deployable_v1.0 from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification
2. pcbclassification: deployable_v1.1 from https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification

## Prerequisites

**Note:** Make sure you are in the root directory (`path/to/inference-builder`) to execute the commands in this README. All relative paths and commands assume you are running from the inference-builder root directory.

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
mkdir ~/.cache/model-repo/
sudo chmod -R 777 ~/.cache/model-repo/
export MODEL_REPO=~/.cache/model-repo/
```

You need first download the model files from the NGC catalog and put them in the $MODEL_REPO/{model-name}/ directory, then copy the other required configurations to the same directory:

### For pcbclassification sample

Please use `pcbclassification` as the directory:

```bash
ngc registry model download-version "nvaie/pcbclassification:deployable_v1.1"
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
mv pcbclassification_vdeployable_v1.1 $MODEL_REPO/pcbclassification
chmod 777 $MODEL_REPO/pcbclassification
cp -r builder/samples/ds_app/classification/pcbclassification/* $MODEL_REPO/pcbclassification/
```

### For changenet-classify sample

Please use `changenet-classify` as the directory:

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_classification:visual_changenet_nvpcb_deployable_v1.0"
mv visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0 $MODEL_REPO/changenet-classify/
# Move the folder to the model-repo directory, and the sample uses ~/.cache/model-repo by default
chmod 777 $MODEL_REPO/changenet-classify
cp -r builder/samples/ds_app/classification/changenet-classify/* $MODEL_REPO/changenet-classify/
```

## Generate the DeepStream Application Package and Build Container Image

### For pcbclassification sample

Please use ds_pcb.yaml as the configuration:

#### For x86 Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    builder/samples/ds_app
```

#### For Tegra Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/classification/ds_pcb.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```

### For changenet-classify sample

Please use ds_changenet.yaml as the configuration:

#### For x86 Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    builder/samples/ds_app
```

#### For Tegra Architecture

```bash
export GITLAB_TOKEN={Your GitLab Token}
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml \
    -o builder/samples/ds_app \
    --server-type serverless \
    -t \
&& docker build \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t deepstream-app \
    -f builder/samples/ds_app/Dockerfile.tegra \
    builder/samples/ds_app
```


## Run the deepstream app with image inputs:

**Note:** You can optionally set the `$SAMPLE_INPUT` environment variable to point to your samples directory if you perform inference on media files in your host.

```bash
# Update this with your actual samples directory path
export SAMPLE_INPUT=/path/to/your/samples/directory
```

### For pcbclassification sample

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
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

### For changenet-classify sample

**Note:** For changenet classification model, it requires two images as input. Here we are using the sample images provided in the sample_input directory as a reference.

```bash
# media-url: the path or URL to the input media.
# mime: the media type (e.g., "video/mp4" or "image/jpeg").
# /sample_input/test_1.jpg and /sample_input/golden_1.jpg are just a placeholder for the images present in $SAMPLE_INPUT directory
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