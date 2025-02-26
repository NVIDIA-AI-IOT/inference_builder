#!/bin/bash

# start NIM on localhost

# Generate the OpenAPI client
bash ../../../tools/openapi-gen.sh

# Validate the generated client
python test_inference.py --image /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.jpg --expected ./expected.sample_720p.json

