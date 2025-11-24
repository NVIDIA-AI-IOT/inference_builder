## Introduction

This example demonstrates how to build an inference pipeline with Nvidia Deepstream MediaExtractor and vLLM Inference backends

## Prerequisites

The sample models including `Cosmos-Reason1-7B` and `Qwen3-VL-30B-A3B-Instruct` can all be downloaded from huggingface (Be sure to have git-lfs installed):

```bash
git clone https://huggingface.co/nvidia/Cosmos-Reason1-7B ~/.cache/model-repo/Cosmos
```

or

```bash
git clone https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct ~/.cache/model-repo/Qwen3-VL
```

## Generate the Inference Package


### Cosmos-Reason1-7B

```bash
python builder/main.py builder/samples/vllm/vllm_nvdec_cosmos.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/vllm/ -c builder/samples/qwen/processors.py -t
```

### Qwen3-VL-30B-A3B-Instruct

```bash
python builder/main.py builder/samples/vllm/vllm_nvdec_qwen3_vl.yaml --api-spec builder/samples/qwen/openapi.yaml \
-o builder/samples/vllm/ -c builder/samples/qwen/processors.py -t
```

## Build and Start the Inference Microservice:

The sample folder already contains all the ingredients for building the microservice, all you need is to run the command:

```bash
cd builder/samples
docker compose up ms-cosmos --build
```

## Test the model with a client

There is an OpenAI client included in the sample for testing, and for image input, you can directly run the client with image path:

```bash
cd builder/samples/qwen
python client.py --images <your_image.jpg>
```

For video input, you need to first upload a test video file.

**⚠️ Important:** **replace the placeholder <your_video.mp4> in below command with an actual file path in your system**.

```bash
export VIDEO_FILE=<your_video.mp4>
curl -X "POST" \
  "http://localhost:8800/v1/files" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$VIDEO_FILE;type=video/mp4"
```

You'll get a response 200 with a json body:
{
  "data": {
    "id": "577a9f11-2b24-4db8-82c8-2601e0c2b6e4",
    "path": "/tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4",
    "size": 3472221,
    "duration": 30000000000,
    "contentType": "video/mp4"
  }
}

Run the client.py with the returned video path:

```bash
python client.py --videos 577a9f11-2b24-4db8-82c8-2601e0c2b6e4?frames=8
```

For testing long videos with automatic chunking, you must use raw json payload with curl by setting accept to "application/x-ndjson":

```bash
curl -X 'POST' \
  'http://localhost:8800/v1/chat/completions' \
  -H 'accept: application/x-ndjson' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "cosmos_v1",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Please describe the image in detail."
        },
        {
          "type": "video",
          "video": {
            "url": "577a9f11-2b24-4db8-82c8-2601e0c2b6e4?frames=8&chunks=8"
          }
        }
      ]
    }
  ],
  "max_tokens": 200
}' -N
```
