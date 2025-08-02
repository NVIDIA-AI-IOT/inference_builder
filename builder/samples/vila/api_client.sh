#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

 set -x

# usage
# ./api_client.sh [stream-option] [url]

function usage() {
  echo "$0 http://ip:port/inference image_file, "
  echo "    ip: the IP address of the container "
  echo "    image_file: path to image file"

}

if [[ $# -lt 2 ]]; then
  usage
  exit 0
fi

url=$1
image=$2

image_b64=$( base64 $image )

echo '{
  "messages": [
    {
      "role": "user",
      "content": "Describe what you see in this image. <img src=\"data:image/jpeg;base64,'${image_b64}'\" />"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.70,
  "seed": 0,
  "stream": false
}' > payload.json

curl -N -X POST $url \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d @payload.json
