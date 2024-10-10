import argparse
import requests, base64
import time
import numpy as np
import os
from PIL import Image

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"

palette = [
  {
    "seg_class": "foreground",
    "rgb": [0, 0, 0],
    "label_id": 0,
    "mapping_class": "foreground"
  },
  {
    "seg_class": "background",
    "rgb": [255, 255, 255],
    "label_id": 1,
    "mapping_class": "background"
  }
]


def main(host , port, files, out_format):
  if not files:
    print("Need the file path for inference")
    return

  # invoke_url = "http://localhost:8001/inference"
  invoke_url = "http://" + host + ":" + port + "/v1/masks"

  file_exts = []
  b64_images = []
  for file in files:
    file_exts.append(os.path.splitext(file)[1][1:])
    with open(file, "rb") as f:
      b64_images.append(base64.b64encode(f.read()).decode())

# assert len(image_b64) < 180_000, \
#   "To upload larger images, use the assets API (see docs)"

  headers = {
    "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
    "Accept": "application/json",
    # "NVCF-ASSET-DIR": "some-temp-dir",
    # "NVCF-FUNCTION-ASSET-IDS": "udjflsjo-jfoisjof-lsdfjofdj"
  }

  payload = {
    "input": [f"data:image/{e};base64,{i}" for e, i in zip(file_exts, b64_images)],
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
    id_color_map = {}
    for p in palette:
        id_color_map[p['label_id']] = p['rgb']
    for mask in output["data"]:
      if out_format == 'integer':
        output_mask = np.array(mask["mask"])
        output = Image.fromarray(output_mask.astype(np.uint8)).convert('P')
        output_palette = np.zeros((len(palette), 3), dtype=np.uint8)
        for c_id, color in id_color_map.items():
            output_palette[c_id] = color
        output.putpalette(output_palette)
        output = output.convert("RGB")
        output.show()
        output.save("mask.png")
      else:
        output_mask = np.frombuffer(base64.b64decode(mask["mask"]), dtype="uint8").tolist()

      print(f"index = {mask['index']}")
      print(f"output_mask = {output_mask}")

  #print(response.json())


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--host", type=str,
                    help= "Server IP Address", default="0.0.0.0")
  parser.add_argument("--port", type=str,
                    help="Server port", default="8000")
  parser.add_argument("--file", type=str, help="File to send for inference", nargs='*', default=None)
  parser.add_argument("--format", type=str, help="Output embedding format, integer or base64",
                      default="integer")

  args = parser.parse_args()
  main(args.host, args.port, args.file, args.format)
