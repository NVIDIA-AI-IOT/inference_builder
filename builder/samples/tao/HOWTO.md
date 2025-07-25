# Metropolis Computer Vision Inference Microservice HOWTO

## Introduction

This document provides instructions for building Metropolis Computer Vision Inference Microservices and using them to perform inference on images and videos.

## Prerequisites

Below packages are required to build and run the microservice:

- Docker
- Docker Compose
- NVIDIA Container Toolkit

## Models used in the Samples

All the models can be downloaded from NGC:

### Image Classification
- **PCB Classification**: [PCB Classification Model](https://catalog.ngc.nvidia.com/orgs/nvaie/models/pcbclassification)

```bash
ngc registry model download-version "nvaie/pcbclassification:deployable_v1.1"
```

### Visual Change Detection
- **Visual Changenet Classification**: [Visual Changenet Classification Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/visual_changenet_classification)

```bash
ngc registry model download-version "nvidia/tao/visual_changenet_classification:visual_changenet_nvpcb_deployable_v1.0"
```

### Semantic Segmentation
- **CitySemSegFormer**: [CitySemSegFormer Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer)

```bash
ngc registry model download-version "nvidia/tao/citysemsegformer:deployable_onnx_v1.0"
```

### Grounding Dino
- **Grounding DINO**: [Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino)
- **Mask Grounding DINO**: [Mask Grounding DINO Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/mask_grounding_dino)

```bash
ngc registry model download-version "nvidia/tao/grounding_dino:grounding_dino_swin_tiny_commercial_deployable_v1.0"
ngc registry model download-version "nvidia/tao/mask_grounding_dino:mask_grounding_dino_swin_tiny_commercial_deployable_v1.0"
```

### Resnet50 RT-DETR Detector
- **Resnet50 RT-DETR**: [To be added]

## Build the microservice

First, follow the instructions in the top-level README to set up Inference Builder. Once the setup is complete, you need to generate the inference package before building the microservice. This can be done using the Inference Builder tool as shown below:

For generic TAO CV models:
```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

For visual changenet:
```bash
python builder/main.py builder/samples/tao/ds_changenet.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

For Grounding DINO and Mask Grounding DINO:
```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -c builder/samples/tao/processors.py -t
```

## Build and Run the microservices using docker compose

A tao-cv service is defined in the pre-configured docker-compose.yml file located in the builder/samples directory. It supports all the above three inference pipelines.

Within the container image, the inference pipeline searches for model files in /workspace/.cache/model_repo. Therefore, the corresponding directory on the host machine must be created and mounted as a volume inside the container.
By default, this directory on the host is $HOME/.cache/model-repo.

Each model’s required files must be placed in a separate subdirectory under model-repo. These files may include the ONNX model, nvinfer configuration files, label files, and other relevant assets. To ensure the inference microservice selects the correct model, the TAO_MODEL_NAME environment variable must be set to the name of the corresponding subdirectory under model-repo.

Create the model-repo if it doesn't exist yet:

```bash
mkdir -p ~/.cache/model-repo
chmod 777 ~/.cache/model-repo
mkdir ~/.cache/model-repo/{TAO_MODEL_NAME}
chmod 777 ~/.cache/model-repo/{TAO_MODEL_NAME}
```

Following files are expected to be present in the model directory:

- Deepstream inference config file : `nvdsinfer_config.yaml`
- ONNX model file
- Label file (optional, used for post-processing)
- preprocessed config file (optional, used for pre-processing)

Configurations for sample tao models can be found from builder/samples/ds_app:

- builder/samples/ds_app/classification/changenet-classify/: Visual Changenet classification model with 2 image inputs
- builder/samples/ds_app/classification/pcbclassification/: PCB classification model
- builder/samples/ds_app/detection/rtdetr: RT-DETR detection model
- builder/samples/ds_app/gdino/gdino: Grounding Dino detection model
- builder/samples/ds_app/gdino/mask_gdino: Mask Grounding Dino detection model

When being used along with the TAO Finetune Microservice, the microservice can directly use the model files and configs exported from Finetune Microservice.

Once the model files are ready, you can set the TAO_MODEL_NAME to the model you want in docker-compose.yml and run the following command to build and start the microservice:

```bash
cd builder/samples
docker compose up tao-cv --build
```

**Note:** Regenerate the inference pipeline for the new model using Inference Builder after you switch the model and rebuild the microservice.

## Use the microservice

The microservice provides a REST API that can be used to run inference on images and videos.

The OpenAPI compatible interactive documentation endpoint is available on the server for detailed API usage: http://localhost:8800/docs, and examples to show the basic inference use cases are listed as below:

### Run inference on a single image

```bash
PAYLOAD=$(echo -n "data:image/jpeg;base64,"$(base64 -w 0 "your_file.jpg"))

curl -X POST \
  'http://localhost:8800/v1/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"input\": [ \"$PAYLOAD\" ],
  \"model\": \"nvidia/nvdino-v2\"
}"

```

### Run inference on a single video

```bash
# upload the video file as an asset
curl -X 'POST' \
  'http://localhost:8800/v1/files' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_file.mp4;type=video/mp4'
```

The expected response would be like:

```
{
  "data": {
    "id": "53c2d620-976e-49a4-90a3-3db20b95d225",
    "path": "/tmp/assets/53c2d620-976e-49a4-90a3-3db20b95d225/output.mp4",
    "size": 82223,
    "duration": 2000000000,
    "contentType": "video/mp4"
  }
}
```

Use the received asset object for inference. ('accept' must be application/x-ndjson to get the inference result as a stream)

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
  "model": "nvidia/nvdino-v2"
}' -N

```

The OpenAPI specification also serves as the guideline for developing customized inference client and integrate it to any application.

## Validate the Inference Microservices

Inference Builder support generate automatic validation script against individual models.

### Build inference package with validation:

Validation script will be generated when `-v <path_to_validation_src>` is appended to the build command.

#### RTDETR detection model

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t \
-v builder/samples/tao/validation/rtdetr
```

After running docker compose as specified in the previous section, run the validation script as follows:

```bash
cd builder/samples/tao/validation/rtdetr/.tmp/ && python test_runner.py
```


#### Image classification model (pcbclassification)

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t \
-v builder/samples/tao/validation/pcbclassification
```

After running docker compose as specified in the previous section, run the validation script as follows:

```bash
cd builder/samples/tao/validation/pcbclassification/.tmp && python test_runner.py
```

#### Segmentation model (citysemsegformer)

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -t \
-v builder/samples/tao/validation/citysemsegformer
```

After running docker compose as specified in the previous section, run the validation script as follows:

```bash
cd builder/samples/tao/validation/citysemsegformer/.tmp && python test_runner.py
```

#### Grounding DINO

```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -c builder/samples/tao/processors.py -t \
-v builder/samples/tao/validation/gdino
```

After running docker compose as specified in the previous section, run the validation script as follows:

```bash
cd builder/samples/tao/validation/gdino/.tmp/ && python test_runner.py
```

#### Mask Grouding DINO

```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml \
-a builder/samples/tao/openapi.yaml -o builder/samples/tao -c builder/samples/tao/processors.py -t \
-v builder/samples/tao/validation/mgdino
```

After running docker compose as specified in the previous section, run the validation script as follows:

```bash
cd builder/samples/tao/validation/mgdino/.tmp && python test_runner.py
```
