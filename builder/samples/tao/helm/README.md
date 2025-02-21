# Prerequisites
1. UCS tools is installed. https://ucf.gitlab-master-pages.nvidia.com/docs/master/text/UCS_Installation.html
2. microk8s is installed

# Build TAO CV App Helm Chart
make -C tao-cv-app


# Prepare models on host path for k8s

## 1. Create storageClass with name "mdx-local-path", using Local Path Provisioner
```
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.23/deploy/local-path-storage.yaml | sed 's/^  name: local-path$/  name: mdx-local-path/g' | microk8s kubectl apply -f -
```

NOTE:
1. A base path will be created at /opt/local-path-provisioner


## 2. Create PVC with name "local-path-pvc", resolving to the created storageClass "mdx-local-path"
```
$ curl https://raw.githubusercontent.com/rancher/local-path-provisioner/master/examples/pvc/pvc.yaml | \
    sed -e 's/storageClassName: local-path$/storageClassName: mdx-local-path/g' \
        -e 's/storage: 128Mi$/storage: 10Gi/g' | \
    microk8s kubectl apply -f -
```

NOTE: 
1. This helm chart defines volume resolving to PVC "local-path-pvc"

2. A sub directory will be created as /opt/local-path-provisioner/pvc-*_default_local-path-pvc

## 3. Copy model files including model, configs to /opt/local-path-provisioner/pvc-*_default_local-path-pvc, following below structure. Make sure the each model sub directory name matches the model name in the custom_values.yaml
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

# Run the TAO CV App helm chart in k8s
Update the custom_values.yaml with the TaskHead model name
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
