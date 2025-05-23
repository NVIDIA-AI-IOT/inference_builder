#!/bin/bash
set -e  # Exit on any error

# Update package list and install required packages
apt-get update
apt-get install -y openssh-client sshpass curl apt-transport-https iputils-ping traceroute net-tools iproute2 netcat-openbsd

# install kubectl
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
apt-get update
apt-get install -y kubectl

# Add Helm repository and install Helm
echo "deb https://baltocdn.com/helm/stable/debian/ all main" | tee /etc/apt/sources.list.d/helm-stable-debian.list
curl https://baltocdn.com/helm/signing.asc | apt-key add -
apt-get update
apt-get install -y helm

# Install Python packages
pip install pyyaml requests