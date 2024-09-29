import argparse
import requests, base64
import time
import numpy as np

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"

def main(host , port, file, out_format):
  if not file:
    print("Need the file path for inference")
    return

  # invoke_url = "http://localhost:8001/inference"
  invoke_url = "http://" + host + ":" + port + "/v1/embeddings"

  with open(file, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# assert len(image_b64) < 180_000, \
#   "To upload larger images, use the assets API (see docs)"

  headers = {
    "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
    "Accept": "application/json",
    # "NVCF-ASSET-DIR": "some-temp-dir",
    # "NVCF-FUNCTION-ASSET-IDS": "udjflsjo-jfoisjof-lsdfjofdj"
  }

  payload = {
    "input": [f"data:image/jpeg;base64,{image_b64}"
    ],
    "encoding_format": out_format,
    "model": "nvidia/nvdinov2-vit-g"
  }
  start_time = time.time()
  response = requests.post(invoke_url, headers=headers, json=payload)
  infer_time = time.time() - start_time
  print(response)
  print(infer_time)

  if response.status_code == 200:
    output = response.json()
    print(f"Usage: num_images= {output['usage']['num_images']}")
    print("Output:")
    for embedding in output["data"]:
      if out_format == 'float':
        output_embedding = embedding["embedding"]
      else:
        embedding_string = embedding["embedding"]
        output_embedding = np.frombuffer(base64.b64decode(embedding_string), dtype="float32").tolist()

      print(f"index = {embedding['index']}")
      print(f"output_embedding = {output_embedding}")

  #print(response.json())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--host", type=str,
                    help= "Server IP Address", default="0.0.0.0")
  parser.add_argument("--port", type=str,
                    help="Server port", default="8000")
  parser.add_argument("--file", type=str, help="File to send for inference",
                      default=None)
  parser.add_argument("--format", type=str, help="Output embedding format, float or base64",
                      default="float")

  args = parser.parse_args()
  main(args.host, args.port, args.file, args.format)
