This example demonstrates how to build an inference pipeline and package it to a NIM for the Qwen family of VLM models using the Inference Builder tool. Following models have been tested:
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct

Three configurations are provided, allowing you to choose based on your specific software and hardware environment:

1. pytorch_qwen.yaml: leveraging the transformer APIs and fits all the models
2. trtllm_qwen.yaml: leveraging TensroRT LLM APIs for better performance
3. trtllm_nvdec_qwen.yaml: leveraging h/w decoder and TensorRT LLM for the best performance

How to Build the inference code:

First you must follow the top level README.md to set up the enviroment, and then run

a. Using baseline pytorch backend via transfomer APIs:

python builder/main.py builder/samples/qwen/pytorch_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml -o builder/samples/qwen/ -c builder/samples/qwen/processors.py --server-type nim -t

b. Using optimized tensorrtllm pytorch backend:

python builder/main.py builder/samples/qwen/trtllm_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml -o builder/samples/qwen/ -c builder/samples/qwen/processors.py --server-type nim -t

OR

python builder/main.py builder/samples/qwen/trtllm_nvdec_qwen.yaml --api-spec builder/samples/qwen/openapi.yaml -o builder/samples/qwen/ -c builder/samples/qwen/processors.py --server-type nim -t

How to Run the inference pipeline within a NIM:

The sample folder already contains all the ingredients for building a NIM, you only need to run

cd builder/samples
docker compose up nim-qwen --build

How to adapt the generated inference package to your own NIM environment:

1. All the prerequisites listed in the dependencies.yaml be copied to your dependencies.
2. Properly set up your base container image:
2.1. If you choose transforms pytorch backend (pytorch_qwen.yaml), you must install the latest transformers(pip install git+https://github.com/huggingface/transformers accelerate) and qwen utils(pip install qwen-vl-utils[decord]==0.0.8) as dependencies.
2.2. If you choose tensorrtllm backend, you must get the latest TensorRT LLM base container(v0.20.0rc3) or build your tensrortllm image and use it as base image.
2.3. If you choose to enable U/W decoder, you must copy and install DeepstreamSDK from the Deepstream 8.0 container to your tensorrtllm base image.
3. Download your model checkpoints and map the parent folder to /opt/nim/.cache/model-repo. You can also use your preferred model folder by modifying the "model_repo" in the yaml configuration file and rebuild the inference package.
4. Add the inference package to the contaienr image (ADD qwen.tgz $NIM_DIR_PATH)
5. If you're using a different OpenAPI specification in your NIM, you must modify the request/response mapping templates accordingly in the yaml configuration and rebuild the inference package.

There is an OpenAI client included in the sample for testing the NIM.

For image input, you can directly run the client with image path:

python client.py --images your_image.jpg


For video input, you need to first upload a test video file:

curl -X 'POST' \
  'http://localhost:8803/v1/files' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@its_1920_30s.mp4;type=video/mp4'

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

python client.py --videos /tmp/assets/577a9f11-2b24-4db8-82c8-2601e0c2b6e4/its_1920_30s.mp4

OR run the client.py with the returned asset id if H/W encoder configuration is used:

python client.py --videos 577a9f11-2b24-4db8-82c8-2601e0c2b6e4?frames=8
