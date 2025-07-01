# Introduction

This sample demonstrates how to build the inference pipeline for VILA 1.5 and how to integrate it into a NIM.

# Prerequisites

## The model repo

We need to prepare a folder to accommodate model data, and we use ~/.cache/nim/model-repo for this sample:

```bash
mkdir -p ~/.cache/nim/model-repo
```

## The model

VILA 1.5 checkpoints can be downloaded from huggingface using the following command:

```bash
git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-13b ~/.cache/nim/model-repo/vila1.5-13b
```

## Optimized engine files

Before generating the inference pipeline, we need the optimized engine files in hand. Please follow the command below to convert above checkpoints and generate optimized engine files:

```bash
docker login nvcr.io
docker run -it --rm --gpus all -v ~/.cache/nim/model-repo/vila:/workspace/checkpoints/optimized -v ~/.cache/nim/model-repo:/workspace/checkpoints/baseline -e LLM_BATCH_SIZE=8 -e LLM_PRECISION=int4_awq nvcr.io/ztko6yjat3l5/vlm/vila_trt_engine:0.6.4
```

Once the above process is completed, we'll have the output under ~/.cache/nim/model-repo/vila:

```
└── vila1.5-13b
    ├── fp16
    │   └── 1-gpu
    │       └── visual_engines
    │           ├── config.json
    │           └── visual_encoder.engine
    ├── int4_awq
    │   └── 1-gpu
    │       ├── config.json
    │       └── rank0.engine
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    └── tokenizer.model
```

The two engine files of visual_encoder.engine and rank0.engine will be used by TensorRT-LLM to accelerate the inference of vision encode and language model respectively

# Build the inference pipeline:

```bash
python builder/main.py --server-type triton builder/samples/vila/tensorrt_vila1.5.yaml -a builder/samples/vila/openapi.json -c builder/samples/vila/processors.py -o builder/samples/vila -t
```


# Build and run the docker image:

```bash
cd builder/samples
docker compose up --build nim-vila
```

# Test with the client:

```bash
cd builder/samples/vila
./api_client.sh http://localhost:8803/inference <PNG or JPG file>
```




