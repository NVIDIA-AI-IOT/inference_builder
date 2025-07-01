#!/bin/bash
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
