#!/bin/bash
set -e

 set -x

# usage
# ./api_client.sh [stream-option] [url]

function usage() {
  echo "$0 [stream_option] http://ip:80003/inference png_file, "
  echo "    stream_option: select [true, false] "
  echo "    ip: the IP address of the conatienr "
  echo "    png_file: path to .png file, default [test.png] "

}

if [[ $# -ge 1 && $1 == "--help" ]]; then
  usage
  exit 0
fi

stream=false

if [ $# -ge 1 ]; then
  stream=$1
fi

url="http://localhost:8803/inference"
if [ $# -ge 2 ]; then
  url=$2
fi

if [ "$stream" = true ]; then
    accept_header='Accept: text/event-stream'
else
    accept_header='Accept: application/json'
fi

image="test.png"
if [ $# -ge 3 ]; then
  image=$3
fi

image_b64=$( base64 $image )

echo '{
  "messages": [
    {
      "role": "user",
      "content": "Describe what you see in this image. <img src=\"data:image/png;base64,'${image_b64}'\" />"
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.20,
  "top_p": 0.70,
  "seed": 0,
  "stream": '$stream'
}' > payload.json

curl -N -X POST $url \
  -H "Content-Type: application/json" \
  -H "$accept_header" \
  -d @payload.json
