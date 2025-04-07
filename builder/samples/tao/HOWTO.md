# Metropolis Computer Vision Microservice HOWTO

## Introduction

This document provides instructions for building the Metropolis Computer Vision microservice and using it to run inference on images and videos.

## Prerequisites

Below packages are required to build and run the microservice:

- Docker
- Docker Compose
- NVIDIA Container Toolkit

## Run the microservice

### 1. Run the microservice using docker compose

There're 3 inference microservices available for different use cases:
- Common CV tasks for classification, detection and segmentation: gitlab-master.nvidia.com:5005/chunlinl/nim-templates/tao_cv_nim:ds8.0-triton25.02.1_2
- Visual Change Net:
- Open Label detection and segmenation:

Create the docker compose file from the below sample and save it as docker-compose.yaml:

```yaml
services:
  tao-cv:
    image: gitlab-master.nvidia.com:5005/chunlinl/nim-templates/tao_cv_nim:ds8.0-triton25.02.1_2
    volumes:
      - '~/.cache/nim:/opt/nim/.cache'
    ipc: host
    # init: true
    ports:
      - "8800-8803:8000-8003"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              #count: 1
              device_ids: ['0']
              capabilities: [gpu]
    environment:
      NIM_MODEL_NAME: rtdetr
```

Before running the microservice, users must set the name of the model in the `docker-compose.yaml` file through the `NIM_MODEL_NAME` environment variable, meanwhile, users need to prepare the model files and drop them into the `~/.cache/nim/model_repo/{NIM_MODEL_NAME}` directory.

Run below commands to create the model directory:

```bash
mkdir -p ~/.cache/nim/model-repo
chmod 777 ~/.cache/nim/model-repo
mkdir ~/.cache/nim/model-repo/{NIM_MODEL_NAME}
chmod 777 ~/.cache/nim/model-repo/{NIM_MODEL_NAME}
```

Following files are expected to be present in the directory:

- Deepstream inference config file : `nvdsinfer_config.yaml`
- ONNX model file
- Label file (optional, used for post-processing)
- preprocessed config file (optional, used for pre-processing)

When being used along with the TAO Finetune Microservice, the microservice can directly use the model files and configs exported from Finetune Microservice.

Once the model files are ready, users can run the following command to start the microservice:

```bash
docker compose up tao-cv

```

### 2. Run the Microservice as Helm Charts

Users can also use the helm charts to deploy different TAO CV models simply by updating env NIM_MODEL_NAME, and there is no need to rebuild the helm charts.
A values override file is provided to set the NIM_MODEL_NAME; and image.repository, image.tag in case there is a new image.

Update the helm/tao-cv-app/custom_values.yaml for:
1. NIM_MODEL_NAME, which is the TaskHead model name. It has to match the sub directory name in /opt/local-path-provisioner/pvc-*_default_local-path-pvc;
2. image.repository, image.tag if necessary.


#### 2.1 Prepare models on host path for k8s

##### 2.1.1 Create storageClass with name "mdx-local-path", using Local Path Provisioner

###### a) Check if storageClass with name "mdx-local-path" exists

```bash
$ microk8s kubectl get sc
```

eg:
```bash
$ microk8s kubectl get sc
NAME                          PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
mdx-local-path                rancher.io/local-path   Delete          WaitForFirstConsumer   false                  141d
microk8s-hostpath (default)   microk8s.io/hostpath    Delete          WaitForFirstConsumer   false                  147d
```

###### b) If not, create storageClass with name "mdx-local-path". Otherwise, skip the following steps.

```bash
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.23/deploy/local-path-storage.yaml | sed 's/^  name: local-path$/  name: mdx-local-path/g' | microk8s kubectl apply -f -
```

NOTE:
1. A base path will be created at /opt/local-path-provisioner. But it won't be available until a PVC is created & the first pod accessing it.

##### 2.1.2 Create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"
###### a) Check if PVC with name "local-path-pvc" exists and is on storageClass "mdx-local-path"

```bash
$ microk8s kubectl get pvc
```

eg:
```bash
$ microk8s kubectl get pvc
NAME            STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
local-path-pvc   Bound    pvc-68509dc3-2f07-4e8f-8298-4a044b59546b   10Gi       RWO            mdx-local-path   141d
```

###### b) If not, create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"

```bash
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/master/examples/pvc/pvc.yaml | \
    sed -e 's/storageClassName: local-path$/storageClassName: mdx-local-path/g' \
        -e 's/storage: 128Mi$/storage: 10Gi/g' | \
    microk8s kubectl apply -f -
```

NOTE:
1. This helm chart defines volume resolving to PVC "local-path-pvc"

2. A sub directory will be created as /opt/local-path-provisioner/pvc-*_default_local-path-pvc. It won't be available until a pod accessing it.

3. You might see status "Pending" for the PVC "local-path-pvc" before the first pod accessing it.

##### 2.1.3 For each model, create a sub directory under the Local Path Provisioner PVC path, and copy model file including model, configs to it.

Remember to grant 777 permission to model path, otherwise generating engine file will hang.

```bash
cd /opt/local-path-provisioner/pvc-*_default_local-path-pvc
mkdir {NIM_MODEL_NAME}
chmod 777 {NIM_MODEL_NAME}
```

Eg:
```bash
## On host at /opt/local-path-provisioner/pvc-*_default_local-path-pvc
├── {NIM_MODEL_NAME}/
│   ├── model.onnx
│   └── config_nvinfer.yml
├── rtdetr/
│   ├── model.onnx
│   └── config_nvinfer.yml
└── segformer/
    ├── model.onnx
    └── config_nvinfer.yml

## What containers can see at /opt/nim/.cache/model-repo/
├── {NIM_MODEL_NAME}/
│   ├── model.onnx
│   └── config_nvinfer.yml
├── rtdetr/
│   ├── model.onnx
│   └── config_nvinfer.yml
└── segformer/
    ├── model.onnx
    └── config_nvinfer.yml
```

Make sure the each model sub directory name {NIM_MODEL_NAME} matches that in the custom_values.yaml

#### 2.2 Create image pull secret, if not yet created
```bash
# create secret docker-registry ngc-docker-reg-secret for pulling containers from nvcr.io
microk8s kubectl create secret docker-registry ngc-docker-reg-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_CLI_API_KEY
```

#### 2.3 Fetch the helm chart from NGC
```bash
$ helm fetch https://helm.ngc.nvidia.com/eevaigoeixww/staging/charts/tao-cv-app-0.0.2.tgz --username='$oauthtoken' --password=<YOUR API KEY>
```

#### 2.4 Start

```bash
$ microk8s helm3 install tao-cv-app tao-cv-app-0.0.2.tgz -f custom_values.yaml
```
OR if you don't want to use the values override file, you can run,
```bash
$ microk8s helm3 install tao-cv-app tao-cv-app-0.0.2.tgz --set tao-cv.applicationSpecs.tao-cv-deployment.containers.tao-cv-container.env[0].value={NIM_MODEL_NAME}
```

#### 2.5 Stop

```bash
microk8s helm3 delete tao-cv-app

```

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

```
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


## Build the microservice

As a prerequisite to build the microservice, the inference package must be generated by inference builder tool as follows:

Build inference package:

```bash
cd ../../..
source .venv/bin/activate
pip install -r requirements.txt
```
For generic TAO CV models:
```bash
python builder/main.py builder/samples/tao/ds_tao.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

For visual changenet:
```bash
python builder/main.py builder/samples/tao/ds_changenet.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

For Grounding DINO and Mask Grouding DINO:
```bash
python builder/main.py builder/samples/tao/ds_gdino.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -o builder/samples/tao - -t
```

Build inference package with validation:

append `-v <path_to_validation_src>` to the command, eg for rtdetr model

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -v builder/samples/tao/validation/rtdetr -o builder/samples/tao -t
```

The microservice uses post-processing to convert the output of the model to the format expected by the Metropolis Computer Vision endpoint, and to fetch and build the post processing library you need to have gitlab token and put it to environment variable `GITLAB_TOKEN`.

```bash
export GITLAB_TOKEN=<your_gitlab_token>
```

To build the microservice, run the following command:

```bash
cd ..
docker compose build nim-tao

```

## Build the helm chart

1. UCS tools is installed. https://ucf.gitlab-master-pages.nvidia.com/docs/master/text/UCS_Installation.html
2. microk8s is installed

### 1. Build TAO CV App Helm Chart
#### 1.1 Build
```bash
cd helm
make -C tao-cv-app
```
#### 1.2 Test (with models PVC configured, follow above steps)
```bash
# Overwrite the container image if necessary.
make -C tao-cv-app install
```
#### 1.3 Stop
```bash
make -C tao-cv-app uninstall
```

### 2.Push helm chart to NGC
#### 2.1 Prerequisites. RUN ONCE ONLY.
```bash
$NGC_API_KEY needs to be present for ngc CLI to work.
$ ngc registry chart create <org_id>/<team_name>/<helm_chart_ngc_page_name> --short-desc <description>
```

#### 2.2 Update chart version
update helm chart version field
```bash
$ vim helm/tao-cv-app/output/Chart.yaml
```
#### 2.3 Package helm chart to .tgz file
```bash
$ cd helm/tao-cv-app/output
$ helm package .
```

#### 2.4 Push the .tgz file to NGC.
```bash
$NGC_API_KEY needs to be present for ngc CLI to work.
$ ngc registry chart push <org_id>/<team_name>/<helm_chart_ngc_page_name>:<new_version>
```

Eg:
```bash
$ ngc registry chart push eevaigoeixww/staging/tao-cv-app:<new_version>
```
