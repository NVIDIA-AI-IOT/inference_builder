image=$1
image_b64=$( base64 $image )

url="http://localhost:8800/v1/embeddings"
echo '{
    "input": ["Image of a dog",
          "data:image/png;base64,'${image_b64}'"
          ],
    "model": "nvidia/nvclip-vit-h-14",
    "encoding_format": "float"
  }' > payload.json

curl -N -X POST $url \
  -H "Content-Type: application/json" \
  -H "$accept_header" \
  -d @payload.json

