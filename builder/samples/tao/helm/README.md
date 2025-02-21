# Prerequisites
1. UCS tools is installed. https://ucf.gitlab-master-pages.nvidia.com/docs/master/text/UCS_Installation.html
2. microk8s is installed

# Build TAO CV App Helm Chart
make -C tao-cv-app


# Prepare models on host path for k8s

## 1. Create storageClass with name "mdx-local-path", using Local Path Provisioner
### 1.1 Check if storageClass with name "mdx-local-path" exists
```
$ microk8s kubectl get sc
```
eg:
```
$ microk8s kubectl get sc
NAME                          PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
mdx-local-path                rancher.io/local-path   Delete          WaitForFirstConsumer   false                  141d
microk8s-hostpath (default)   microk8s.io/hostpath    Delete          WaitForFirstConsumer   false                  147d
```


### 1.2 If not, create storageClass with name "mdx-local-path". Otherwise, skip the following steps.
```
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.23/deploy/local-path-storage.yaml | sed 's/^  name: local-path$/  name: mdx-local-path/g' | microk8s kubectl apply -f -
```

NOTE:
1. A base path will be created at /opt/local-path-provisioner. But it won't be available until a PVC is created & the first pod accessing it.


## 2. Create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"
### 2.1 Check if PVC with name "local-path-pvc" exists and is on storageClass "mdx-local-path"
```
$ microk8s kubectl get pvc
```
eg:
```
$ microk8s kubectl get pvc
NAME            STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
local-path-pvc   Bound    pvc-68509dc3-2f07-4e8f-8298-4a044b59546b   10Gi       RWO            mdx-local-path   141d
```

### 2.2 If not, create PVC with default name "local-path-pvc", on the created storageClass "mdx-local-path"
```
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/master/examples/pvc/pvc.yaml | \
    sed -e 's/storageClassName: local-path$/storageClassName: mdx-local-path/g' \
        -e 's/storage: 128Mi$/storage: 10Gi/g' | \
    microk8s kubectl apply -f -
```

NOTE: 
1. This helm chart defines volume resolving to PVC "local-path-pvc"

2. A sub directory will be created as /opt/local-path-provisioner/pvc-*_default_local-path-pvc. It won't be available until a pod accessing it.

3. You might see status "Pending" for the PVC "local-path-pvc" before the first pod accessing it.

## 3. Copy model files including model, configs to /opt/local-path-provisioner/pvc-*_default_local-path-pvc, following below structure. Make sure the each model sub directory name matches the model name in the custom_values.yaml
Eg:
```
# On host at /opt/local-path-provisioner/pvc-*_default_local-path-pvc
├── rtdetr/
│   ├── model.onnx
│   └── config_nvinfer.yml
└── segformer/
    ├── model.pt
    └── config_nvinfer.yml

# What containers can see at /opt/nim/.cache/model-repo/
├── rtdetr/
│   ├── model.onnx
│   └── config_nvinfer.yml
└── segformer/
    ├── model.onnx
    └── config_nvinfer.yml
```
Then NIM_MODEL_NAME value in custom_values.yaml can be rtdetr or segformer.

# Run the TAO CV App helm chart in k8s
Update the helm/tao-cv-app/custom_values.yaml for:
    1. NIM_MODEL_NAME, which is the TaskHead model name. It need to match the sub directory name in /opt/local-path-provisioner/pvc-*_default_local-path-pvc
    2. image.repository, image.tag if necessary.

```
make -C tao-cv-app install
```

which executes the following command for you:
```
microk8s helm3 install tao-cv-app output -f custom_values.yaml
```

OR equivalently,
```
microk8s helm3 install tao-cv-app output --set tao-cv.applicationSpecs.tao-cv-deployment.containers.tao-cv-container.env[0].value=model_dir_name
```

# Stop the TAO CV App helm chart in k8s
```
make -C tao-cv-app uninstall
```

which executes the following command for you:
```
microk8s helm3 delete tao-cv-app
```
