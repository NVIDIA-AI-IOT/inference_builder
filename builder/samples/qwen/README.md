## Introduction

This example demonstrates how to build an inference pipeline and for the Qwen family of VLM models using the Inference Builder tool. Following models have been tested:
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct

Three configurations are provided, allowing you to choose based on your specific software and hardware environment:

1. pytorch_qwen.yaml: leveraging the transformer APIs and fits all the models
2. trtllm_qwen.yaml: leveraging TensroRT LLM APIs for better performance
3. trtllm_nvdec_qwen.yaml: leveraging h/w decoder and TensorRT LLM for the best performance

## Prerequisites

The model checkpoints can be downloaded from huggingface (Be sure to have git-lfs installed):

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct  ~/.cache/model-repo/Qwen2.5-VL-7B-Instruct
```

## Build and Test the Qwen2.5-VL-7B-Instruct Inference Microservice

### Using baseline pytorch backend via transformer APIs:

#### Generate the Inference Pipeline

```bash
python builder/main.py builder/samples/qwen/pytorch_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

#### Build and Start the Inference Microservice:

The sample folder already contains all the ingredients for building the microservice, all you need is to run the command:

```bash
cd builder/samples && docker compose up ms-qwen --build
```

#### Test the Inference Microservice with a client

Wait about 10 seconds for the server to start, then open a new terminal in your inference-builder folder. The sample includes a test OpenAI client. For image input, run the client with the path to an image file.

```bash
source .venv/bin/activate && cd builder/samples/qwen
python client.py --images <your_image.jpg> # replace placeholder <your_image.jpg> with a true file
```

For video input, you need to first upload a test video file:

```bash
export VIDEO_FILE=<your_video.mp4>
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

You'll get a response 200 with a json body like below:
{
  "data": {
    "id": "577a9f11-2b24-4db8-82c8-2601e0c2b6e4",
    "path": "/tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  }
}

Run the client.py with the video path from your response:

```bash
python client.py --videos /tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4
```

### Using optimized tensorrtllm pytorch backend:

#### Generate the Inference Pipeline

```bash
python builder/main.py builder/samples/qwen/trtllm_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

OR

```bash
# Enable hardware decoder
python builder/main.py builder/samples/qwen/trtllm_nvdec_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/qwen/ -c builder/samples/qwen/processors.py -t
```

#### Build and Start the Inference Microservice:

**Note:** **Before building the docker image for TensorRT-LLM backend, you must comment out line 48 in `./Dockerfile` as below**.

```
line47: # Comment out the following lines if using local built trtllm image
line48: # RUN --mount=type=cache,target=/root/.cache/pip pip install git+https://github.com/huggingface/transformers accelerate
```

And run the commands:

```bash
cd builder/samples && docker compose up ms-qwen --build
```

#### Test the Inference Microservice with a client

Wait about 20 seconds for the server to start, then open a new terminal in your inference-builder folder. The sample includes a test OpenAI client. For image input, run the client with the path to an image file.


```bash
source .venv/bin/activate && cd builder/samples/qwen
python client.py --images <your_image.jpg> # replace placeholder <your_image.jpg> with a true file
```

For video input, you need to first upload a test video file:

```bash
export VIDEO_FILE=<your_video.mp4>
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

You'll get a response 200 with a json body like below:
{
  "data": {
    "id": "577a9f11-2b24-4db8-82c8-2601e0c2b6e4",
    "path": "/tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  }
}

Run the client.py with the video path from your response:

```bash
python client.py --videos /tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4
```

OR run the client.py with the returned asset id if the inference pipeline is built from trtllm_nvdec_qwen.yaml with H/W decoder enabled:

```bash
python client.py --videos 577a9f11-2b24-4db8-82c8-2601e0c2b6e4?frames=8
```