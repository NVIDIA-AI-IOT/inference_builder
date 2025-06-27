# Setup microk8s on remote deployment host
## Install MicroK8s
```bash
sudo snap install microk8s --classic
```

## Add your user to microk8s group
```bash
# eg, user byin
sudo usermod -a -G microk8s byin
sudo chown -R byin ~/.kube
```

## Enable required addons
```bash
microk8s enable dns
microk8s enable gpu  # This automatically handles GPU access
microk8s enable helm3
microk8s enable registry
```

# Get kubeconfig
```bash
mkdir -p ~/.kube
microk8s config > ~/.kube/config
```

# Setup SSH Access
## Onetime only. On your local machine, generate SSH key if you don't have one
```bash
ssh-keygen -t rsa -b 4096 -C "gitlab-deploy"
```

## Copy the private key content - you'll need this for GitLab CI/CD variable
```bash
cat ~/.ssh/id_rsa
```

# Copy public key to remote depoyment host
```bash
# eg, user byin's A6000 host
ssh-copy-id byin@10.111.53.46
```

# Optional: Test the setup
## On remote host, check if GPU is available to MicroK8s
```bash
microk8s kubectl describe node | grep -i gpu
```

## Check if pods can access GPU
```bash
microk8s kubectl run gpu-test --image=nvidia/cuda:11.0-base --command -- nvidia-smi
microk8s kubectl logs gpu-test
```