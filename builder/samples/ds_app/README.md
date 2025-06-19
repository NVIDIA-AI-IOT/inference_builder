This sample demonstrates how to build a deepstream application using Inference Builder:

We have provided sample configs for the following models:

- Image Classification: https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification
- Visual Changenet Classification: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification
- Semantic Segmentation: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer
- Object Detection:
- Mask Grounding Dino: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino
- Grounding Dino: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino

Model files are loaded from '/workspace/models/{MODEL_NAME}' within the container, thus the volume must be correctly mapped from the host.
You need to export MODEL_REPO environment variable to the path where you want to store the model files.

```bash
export MODEL_REPO=/path/to/your/model/repo
```

For example: if you define a model with name "resnet", you must put all the model files include nvconfig, onnx, etc. to a single directory and map it to '/workspace/models/resnet' for the model to be correctly loaded.
Download the model files from the NGC catalog and copy the model onnx file to the '/workspace/models' directory.

For example for object detection model:
Download the model files from the NGC catalog and put them in the $MODEL_REPO/resnet/ directory
To copy the other required model files to the $MODEL_REPO/resnet/ directory, run the following command:

```bash
cp builder/samples/ds_app/detection/resnet/* $MODEL_REPO/resnet/
```

## Generate the application package and build it into a docker image:

### Build the deepstream application package for all models except grounding dino

```bash
python builder/main.py builder/samples/ds_app/detection/ds_detect.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

For any model, replace the model repo with the path to the model builder/samples/ds_app/detection/ds_detect.yaml
For example: for classification model, replace the model repo with the path to the model builder/samples/ds_app/classification/ds_changenet.yaml

```bash
python builder/main.py builder/samples/ds_app/classification/ds_changenet.yaml -o builder/samples/ds_app --server-type serverless -t \
&& docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t deepstream-app builder/samples/ds_app
```

### Build the deepstream application package with grounding dino model

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

### Run with grounding dino model (requires text input)

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

### Run with image input

**Note:** You need to set the `$SAMPLE_INPUT` environment variable to point to your samples directory.

```bash
export SAMPLE_INPUT=/path/to/your/image/directory
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

### Run with changenet classification model (requires two images)

For changenet classification model, it requires two images as input. Here we are using the sample images provided in the sample_input directory as a reference. Be sure to replace the $SAMPLE_INPUT directory with your own images.

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

