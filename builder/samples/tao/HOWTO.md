# Metropolis Computer Vision Inference Microservice HOWTO

## Introduction

This example demonstrates how to Metropolis Computer Vision Inference Microservices with Inference Builder and using them to perform inference on images and videos.

While the sample supports Ampere, Hopper, and Blackwell architectures, the model and the backend set the real hardware requirements.

## Prerequisites

Below packages are required to build and run the microservice:

- Docker
- Docker Compose
- NVIDIA Container Toolkit

## Models used in the Samples

The models used in the example can all be found in NGC and certain models need active subscription.

If you don't have NGC CLI installed, please download and install it from [this page](https://org.ngc.nvidia.com/setup/installers/cli).

### Image Classification
- **PCB Classification**: [PCB Classification Model](https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification)

### Visual Change Classification
- **Visual Changenet Classification**: [Visual Changenet Classification Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification)

### Semantic Segmentation
- **CitySemSegFormer**: [CitySemSegFormer Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer)

### Grounding Dino
- **Grounding DINO**: [Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino)
- **Mask Grounding DINO**: [Mask Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino)

### Resnet50 RT-DETR Detector
- **Resnet50 RT-DETR**: [TrafficCamNet Lite](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet_transformer_lite)

Required configuration files can be found in the deepstream sample folder:

- builder/samples/ds_app/classification/changenet-classify/: Visual Changenet classification model with 2 image inputs
- builder/samples/ds_app/classification/pcbclassification/: PCB classification model
- builder/samples/ds_app/detection/rtdetr: RT-DETR detection model
- builder/samples/ds_app/gdino/gdino: Grounding Dino detection model
- builder/samples/ds_app/gdino/mask_gdino: Mask Grounding Dino detection model

When being used along with the TAO Finetune Microservice, the microservices can directly use the model files and configs exported from [TAO Deploy](https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/tao_deploy_overview.html).

## Build CV Inference Microservices

All three CV inference microservices in the example are built the same way; the only differences are their configurations and processors.

Using the same steps shown in this example, you can also build the CV inference microservice with fine-tuned models exported from [TAO Deploy](https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/tao_deploy_overview.html).

### Generic TAO CV Inference Microservice

This microservice supports common CV models including image classification, object detection and segmentation.

1. Generate the inference pipeline using inference builder (Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment.)

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

2. Download the model files from NGC and apply the configurations. (here, we use object detection model `trafficcamnet_transformer_lite` as an example)

```bash
mkdir -p ~/.cache/model-repo
ngc registry model download-version "nvidia/tao/trafficcamnet_transformer_lite:deployable_v1.0" --dest ~/.cache/model-repo
chmod 777 ~/.cache/model-repo/trafficcamnet_transformer_lite_vdeployable_v1.0
export TAO_MODEL_NAME=trafficcamnet_transformer_lite_vdeployable_v1.0
cp builder/samples/ds_app/detection/rtdetr/* ~/.cache/model-repo/$TAO_MODEL_NAME/
```

3. Build and run the container image

```bash
export GITLAB_TOKEN={Your GITLAB token} # replace the placeholder {Your GITLAB token} with your gitlab token
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

4. Test the microservice

The microservice provides a REST API that can be used to run inference on images and videos.

The OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs, and examples to show the basic inference use cases are listed as below:

- Run inference on a single image

**⚠️ Important:** **replace the placeholder "absolute_path_to_your_file.jpg" in below command with an actual file with absolute path in your system**.

```bash
PAYLOAD=$(echo -n "data:image/jpeg;base64,"$(base64 -w 0 "absolute_path_to_your_file.jpg"))

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"input\": [ \"$PAYLOAD\" ],
  \"model\": \"nvidia/tao\"
}"
```

- Run inference on a single video

**⚠️ Important:** **replace the placeholder <your_video.mp4> in below command with an actual file path in your system**.

```bash
export VIDEO_FILE=<your_video.mp4> # replace the placeholder <your_video.mp4> with an actual file
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

The expected response would be like:

{
  "data": {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  }
}

Now you can invoke the inference API based on the data object in the above response.

**⚠️ Important:** **use the content of data object as the single element of the input list in the payload**.

```base
curl -X 'POST' \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [ {
    "id": "b6eccf7e-0758-4bba-9e51-d6f65b6794f5",
    "path": "/tmp/assets/b6eccf7e-0758-4bba-9e51-d6f65b6794f5/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  } ],
  "model": "nvidia/tao"
}' -N
```


### TAO CV Inference Microservice for Grounding Dino

This microservice supports Grounding Dino model and Mask Grounding Dino model

1. Generate the inference pipeline using inference builder (Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment.)

```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -c builder/samples/tao/processors.py -t
```

2. Download the model files from NGC and apply the configurations. (here, we use Grounding Dino model as an example)

```bash
mkdir -p ~/.cache/model-repo
ngc registry model download-version "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0" --dest ~/.cache/model-repo
chmod 777 ~/.cache/model-repo/grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0
export TAO_MODEL_NAME=grounding_dino_vgrounding_dino_swin_tiny_commercial_deployable_v1.0
cp builder/samples/ds_app/gdino/gdino/* ~/.cache/model-repo/$TAO_MODEL_NAME/
```

3. Build and run the container image

```bash
export GITLAB_TOKEN={Your GITLAB token} # replace the placeholder {Your GITLAB token} with your gitlab token
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

4. Test  the microservice

The microservice provides a REST API that can be used to run inference on images and videos.

The OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs, and examples to show the basic inference use cases are listed as below:

- Run inference on a single image

**⚠️ Important:** **replace the placeholder "path_to_your_file.jpg" in below command with an actual file path in your system**.

```bash
PAYLOAD=$(echo -n "data:image/jpeg;base64,"$(base64 -w 0 "path_to_your_file.jpg"))

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"input\": [ \"$PAYLOAD\" ],
  \"text\": [ [\"car\", \"people\"] ],
  \"model\": \"nvidia/tao\"
}"
```

- Run inference on a single video

**⚠️ Important:** **replace the placeholder <your_video.mp4> in below command with an actual file path in your system**.

```bash
export VIDEO_FILE=<your_video.mp4> # replace the placeholder <your_video.mp4> with an actual file
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

The expected response would be like:

{
  "data": {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  }
}

Now you can invoke the inference API based on the data object in the above response.

**⚠️ Important:** **use the content of data object as the single element of the input list in the payload**.


```base
curl -X 'POST' \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": [ {
    "id": "b6eccf7e-0758-4bba-9e51-d6f65b6794f5",
    "path": "/tmp/assets/b6eccf7e-0758-4bba-9e51-d6f65b6794f5/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  } ],
  "text": [
    ["car", "people"]
  ],
  "model": "nvidia/tao"
}' -N
```
### TAO CV Inference Microservice for Changenet Classification

This microservice supports Visual Changenet Classification model.

1. Generate the inference pipeline using inference builder (Assume you've followed the [top level instructions](../../../README.md#getting-started) to set up the environment.)

```bash
python builder/main.py builder/samples/tao/ds_changenet.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

2. Download the model files from NGC and apply the configurations.

```bash
mkdir -p ~/.cache/model-repo
ngc registry model download-version "nvidia/tao/visual_changenet_classification:visual_changenet_nvpcb_deployable_v1.0" --dest ~/.cache/model-repo
chmod 777 ~/.cache/model-repo/visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0
export TAO_MODEL_NAME=visual_changenet_classification_vvisual_changenet_nvpcb_deployable_v1.0
cp builder/samples/ds_app/classification/changenet-classify/* ~/.cache/model-repo/$TAO_MODEL_NAME/
```

3. Build and run the container image

```bash
export GITLAB_TOKEN={Your GITLAB token} # replace the placeholder {Your GITLAB token} with your gitlab token
cd builder/samples
sed -i "s/TAO_MODEL_NAME: .*/TAO_MODEL_NAME: $TAO_MODEL_NAME/" docker-compose.yml
docker compose up tao-cv --build
```

4. Test the microservice.

Changenet Classification model detects if a part is missing by comparing the test image against a golden image.

Open the a new terminal and go the inference-builder folder, run the commands from your console:

```bash
GOLDEN_PAYLOAD=$(echo -n "data:image/png;base64,"$(base64 -w 0 "builder/samples/tao/IMG_0002_C75.png"))
TEST_PAYLOAD=$(echo -n "data:image/png;base64,"$(base64 -w 0 "builder/samples/tao/IMG_0002_C71.png"))

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"input\": [ \"$GOLDEN_PAYLOAD\", \"$TEST_PAYLOAD\" ],
  \"model\": \"nvidia/tao\"
}"
```
