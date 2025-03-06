# Metropolis Computer Vision Microservice HOWTO

## Introduction

This document provides instructions for building the Metropolis Computer Vision microservice and using it to run inference on images and videos.

## Prerequisites

Below packages are required to build and run the microservice:

- Docker
- Docker Compose
- NVIDIA Container Toolkit

## Run the microservice using docker compose

Before running the microservice, users must set the name of the model can in the `docker-compose.yaml` file through the `NIM_MODEL_NAME` environment variable, meanwhile, users need to prepare the model files and drop them into the `~/.cache/nim/model_repo/{NIM_MODEL_NAME}` directory.

Following files are expected to be present in the directory:

- Deepstream inference config file : `config_nvinfer.yaml`
- ONNX model file
- Label file (optional, used for post-processing)

Once the model files are ready, users can run the following command to start the microservice:

```bash
cd ..
docker compose up tao-cv

```

# Run the Microservice as Helm Charts

Users can use the helm charts to deploy different TAO CV models simply by updating env NIM_MODEL_NAME, and there is no need to rebuild the helm charts.
A values override file is provided to set the NIM_MODEL_NAME; and image.repository, image.tag in case there is a new image.

Update the helm/tao-cv-app/custom_values.yaml for:
1. NIM_MODEL_NAME, which is the TaskHead model name. It has to match the sub directory name in /opt/local-path-provisioner/pvc-*_default_local-path-pvc;
2. image.repository, image.tag if necessary.

## Build TAO CV App Helm Chart

1. UCS tools is installed. https://ucf.gitlab-master-pages.nvidia.com/docs/master/text/UCS_Installation.html
2. microk8s is installed

## Build TAO CV App Helm Chart

```bash
cd helm
make -C tao-cv-app
```

## Prepare models on host path for k8s

### 1. Create storageClass with name "mdx-local-path", using Local Path Provisioner

#### 1.1 Check if storageClass with name "mdx-local-path" exists

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

#### 1.2 If not, create storageClass with name "mdx-local-path". Otherwise, skip the following steps.

```bash
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.23/deploy/local-path-storage.yaml | sed 's/^  name: local-path$/  name: mdx-local-path/g' | microk8s kubectl apply -f -
```

NOTE:
1. A base path will be created at /opt/local-path-provisioner. But it won't be available until a PVC is created & the first pod accessing it.

### 2. Create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"
##### 2.1 Check if PVC with name "local-path-pvc" exists and is on storageClass "mdx-local-path"

```bash
$ microk8s kubectl get pvc
```

eg:
```bash
$ microk8s kubectl get pvc
NAME            STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
local-path-pvc   Bound    pvc-68509dc3-2f07-4e8f-8298-4a044b59546b   10Gi       RWO            mdx-local-path   141d
```

#### 2.2 If not, create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"

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

### 3. For each model, create a sub directory under the Local Path Provisioner PVC path, and copy model file including model, configs to it.

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

## Create image pull secret, if not yet created
```bash
# create secret docker-registry ngc-docker-reg-secret for pulling containers from nvcr.io
microk8s kubectl create secret docker-registry ngc-docker-reg-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_CLI_API_KEY
```


## Run the TAO CV App helm chart in k8s

```bash
make -C tao-cv-app install
```

which executes the following command for you:

```bash
microk8s helm3 install tao-cv-app output -f custom_values.yaml
```

OR equivalently you can also run,

```bash
microk8s helm3 install tao-cv-app output --set tao-cv.applicationSpecs.tao-cv-deployment.containers.tao-cv-container.env[0].value={NIM_MODEL_NAME}
```

## Stop the TAO CV App helm chart in k8s
```bash
make -C tao-cv-app uninstall
```

which executes the following command for you:
```bash
microk8s helm3 delete tao-cv-app

```

## Use the microservice

The microservice provides a REST API that can be used to run inference on images and videos.

### Run inference on an image

A sample client is available as nim_client.py, which follows the OpenAPI specification and can be used as a reference for building your own client.

```bash
python nim_client.py --port 8800 --file <path_to_image>

```

## Build the microservice

As a prerequisite to build the microservice, the inference package must be generated by inference builder tool as follows:

Build inference package:

```bash
cd ../../..
source .venv/bin/activate
pip install -r requirements.txt
python builder/main.py builder/samples/tao/ds_tao.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -o builder/samples/tao -t
```

Build inference package with validation:

append `-v <path_to_validation_src>` to the command, eg for rtdetr model

```bash
python builder/main.py builder/samples/tao/ds_tao.yaml --server-type fastapi -a builder/samples/tao/openapi.yaml -v builder/samples/tao/validation/rtdetr -o builder/samples/tao -t
```

A model-repo folder needs to be created for model drop-in:

```bash
mkdir -p ~/.cache/nim/model-repo
chmod 777 ~/.cache/nim/model-repo
mkdir ~/.cache/nim/model-repo/{NIM_MODEL_NAME}
chmod 777 ~/.cache/nim/model-repo/{NIM_MODEL_NAME}
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

## Push new version of helm chart to NGC
### 0. Prerequisites. RUN ONCE ONLY.
```bash
$NGC_API_KEY needs to be present for ngc CLI to work.
$ ngc registry chart create <org_id>/<team_name>/<helm_chart_ngc_page_name> --short-desc <description>
```

### 1. Update chart version
update helm chart version field
```bash
$ vim helm/tao-cv-app/output/Chart.yaml
```
### 2. Package helm chart to .tgz file
```bash
$ cd helm/tao-cv-app/output
$ helm package .
```

### 3. Push the .tgz file to NGC.
```bash
$NGC_API_KEY needs to be present for ngc CLI to work.
$ ngc registry chart push <org_id>/<team_name>/<helm_chart_ngc_page_name>:<new_version>
```

Eg:
```bash
$ ngc registry chart push eevaigoeixww/staging/tao-cv-app:<new_version>
```

### 4. Fetch the helm chart from NGC
```bash
$ helm fetch https://helm.ngc.nvidia.com/eevaigoeixww/staging/charts/tao-cv-app-0.0.1.tgz --username='$oauthtoken' --password=<YOUR API KEY>
```

### 5. Install the helm chart pulled from NGC
If you want to skip the build steps for container image and helm chart, you can install the helm chart pulled from NGC directly. However, you still need to prepare the k8s resources including PVC, secret, etc mention above.

For running different TAO CV models, you can update the NIM_MODEL_NAME in helm/tao-cv-app/custom_values.yaml

```bash
$ microk8s helm3 install tao-cv-app tao-cv-app-0.0.1.tgz -f custom_values.yaml
```
OR if you don't want to use the values override file, you can run,
```bash
$ microk8s helm3 install tao-cv-app tao-cv-app-0.0.1.tgz --set tao-cv.applicationSpecs.tao-cv-deployment.containers.tao-cv-container.env[0].value={NIM_MODEL_NAME}
```
