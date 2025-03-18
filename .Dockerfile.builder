# Dockerfile.builder
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    git \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use NVIDIA's PyPI
ENV PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple

# Install NIM tools and other Python dependencies
RUN pip install --no-cache-dir \
    nimtools==1.0.0 \
    openapi-generator-cli

# Install project dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working directory
WORKDIR /workspace
