#!/bin/bash

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: $1 not found!"
        exit 1
    fi
}

# Check and read SSH key
SSH_KEY_FILE="$HOME/.ssh/id_rsa"
check_file "$SSH_KEY_FILE"

# Export all required environment variables
export CI_PROJECT_DIR="/inference-builder"
export CI_REGISTRY_IMAGE="gitlab-master.nvidia.com:5005/deepstreamsdk/inference-builder"
export CI_COMMIT_SHA="local-test"
# FLAVOR: tao, gdino
export FLAVOR="tao"
export OVERRIDE_IMAGE="gitlab-master.nvidia.com:5005/deepstreamsdk/inference-builder/cv-tao-tao:20250423.test.2"
export VALIDATOR_IMAGE="gitlab-master.nvidia.com:5005/deepstreamsdk/inference-builder/cv-tao-validation:4a51f955a7ad1038f53fec2f252f53737e171e0d"
export DEPLOYMENT_HOST_IP="10.111.53.46"
export DEPLOYMENT_HOST_USER="byin"
export HELM_CHART_URL="https://helm.ngc.nvidia.com/eevaigoeixww/dev/charts/tao-cv-app-0.1.1.tgz"
export NGC_API_KEY=$NGC_API_KEY
# Read and export SSH key content
export SSH_PRIVATE_KEY=$(cat $SSH_KEY_FILE)

# Print environment summary
echo "Environment variables set:"
echo "------------------------"
echo "CI_PROJECT_DIR: $CI_PROJECT_DIR"
echo "CI_REGISTRY_IMAGE: $CI_REGISTRY_IMAGE"
echo "CI_COMMIT_SHA: $CI_COMMIT_SHA"
echo "FLAVOR: $FLAVOR"
echo "OVERRIDE_IMAGE: $OVERRIDE_IMAGE"
echo "DEPLOYMENT_HOST_IP: $DEPLOYMENT_HOST_IP"
echo "DEPLOYMENT_HOST_USER: $DEPLOYMENT_HOST_USER"
echo "HELM_CHART_URL: $HELM_CHART_URL"
echo "SSH_PRIVATE_KEY: [HIDDEN]"
echo "------------------------"
echo "Environment ready for testing!"




# Set up variables for easier testing
KUBECONFIG=/inference-builder/kubeconfig
NAMESPACE=default
RELEASE=tao-cv

# Check all resources with the instance label
echo "All resources with instance label:"
kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get all -l app.kubernetes.io/instance=$RELEASE

# Check deployments
echo -e "\nChecking deployment:"
kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get deployment -l app.kubernetes.io/instance=$RELEASE

# Get validator pod (using job labels)
echo -e "\nValidator pod:"
kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get pods -l job-name=tao-cv-validator-validator-deployment

# Get logs from validator pod
echo -e "\nValidator logs:"
VALIDATOR_POD=$(kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE get pods -l job-name=tao-cv-validator-validator-deployment -o jsonpath='{.items[0].metadata.name}')
kubectl --kubeconfig $KUBECONFIG -n $NAMESPACE logs $VALIDATOR_POD