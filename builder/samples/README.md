# Introduction

The samples in this folder demonstrates how to use Inference Builder to create inference pipelines with various backend, such as Deepstream, TensorRT, Triton, TensorRT-LLM, etc.With the provided Dockerfile, you can package the generated pipeline into a container image and run it as a standalone app or a microservice. Build steps vary by backend. See the README.md in each example folder exact instructions.

For examples that run as microservices, we’ve provided an all-in-one docker-compose.yml to manage them together. You can customize the container behavior by changing the configurations there accordingly.

While Inference Builder works with Ampere, Hopper, and Blackwell architectures, the examples’ model and backend choices set the real hardware requirements. For example, Qwen2.5-7B-Instruct model with TensorRT-LLM backend requires very high GPU memory and can only run on H100 and B200.

# Prerequisite

Below packages are required to build and run the container images:

- **Docker**: [Installation Guide](https://docs.docker.com/desktop/setup/install/linux/)
- **Docker Compose**: [Installation Guide](https://docs.docker.com/desktop/setup/install/linux/)
- **NVIDIA Container Toolkit**: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


Ensure nvidia runtime added to `/etc/docker/daemon.json` to run GPU-enabled containers

```bash
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

The `docker` group must exist in your system, please check if it has been created using `getent group docker`. Your current user must belong to the `docker` group, If not, run the command below, then log out and back in for the group change to take effect.

```bash
sudo usermod -aG docker $USER
```

# Models

Some of the models for examples are from the NVIDIA GPU Cloud (NGC) repository, and certain models from NGC require active subscription. Please download and install the NGC CLI from the [NGC page](https://org.ngc.nvidia.com/setup/installers/cli) and follow the [NGC CLI Guide] (https://docs.ngc.nvidia.com/cli/index.html) to set up the tool.

# List of Examples:

## Deepstream Backend

- [ds_app](./ds_app/): examples of building standalone deepstream application.
- [tao](./tao/): examples of building inference microservices using deepstream pipeline and fastapi

## Triton Backend

- [changenet](./changenet/): example of building inference microservices with triton server

## TensorRT Backend

- [nvclip](./nvclip/): example of building inference microservices with TensorRT backend

## TensorRT-LLM Backend

- [qwen](./qwen/): example of building inference microservices with TensorRT-LLM backend for vlm models
- [vila](./vila/): example of building infernece microservices with TensorRT-LLM using the legacy flow (to be deprecated)

## Dry-run

- [dummy](./dummy/): This is an example of building dry-run inference pipeline to test the data flow.
