import argparse
import requests, base64
import time
import os
from pyservicemaker import Probe, Pipeline, Flow, BatchMetadataOperator, osd
from pyservicemaker.flow import _parse_stream_info
from typing import List

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 544

class BBoxMarker(BatchMetadataOperator):
    def __init__(self, bboxes: List):
       super().__init__()
       self._bboxes = bboxes

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            bboxes = self._bboxes[frame_meta.pad_index]
            display_meta = batch_meta.acquire_display_meta()
            for b in bboxes:
              x = b[0] * NETWORK_WIDTH
              y = b[1] * NETWORK_HEIGHT
              w = b[2] * NETWORK_WIDTH
              h = b[3] * NETWORK_HEIGHT
              bbox = osd.Rect()
              bbox.left = x - w/2
              bbox.top = y - h/2
              bbox.width = w
              bbox.height = h
              bbox.border_width = 1
              bbox.border_color = osd.Color(1.0, 1.0, 1.0, 1.0)
              display_meta.add_rect(bbox)
            frame_meta.append(display_meta)

def freeze(flow):
  flow._pipeline.add("imagefreeze", "freezer")
  for stream in flow._streams:
      stream_info = _parse_stream_info(stream)
      flow._pipeline.link(stream_info.originator, "freezer")
  return Flow(flow._pipeline, streams=['freezer/src'], parent=flow)

Flow.freeze = freeze

def main(host , port, files, out_format):
  if not files:
    print("Need the file path for inference")
    return

  # invoke_url = "http://localhost:8001/inference"
  invoke_url = "http://" + host + ":" + port + "/v1/infer"

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
    "model": "nvidia/rtdetr"
  }

  start_time = time.time()
  response = requests.post(invoke_url, headers=headers, json=payload)
  infer_time = time.time() - start_time
  print(response)
  print(infer_time)

  bboxes_list = []
  if response.status_code == 200:
    output = response.json()
    print(f"Usage: num_images= {output['usage']['num_images']}")
    print("Output:")
    for data in output["data"]:
      bboxes = data["bboxes"]
      probs = data["probs"]
      print(f"index = {data['index']}")
      print(f"bboxes = {bboxes}")
      print(f"probs = {probs}")
      bboxes_list.append(bboxes)

  pipeline = Pipeline('renderer')
  bbox_marker = BBoxMarker(bboxes_list)
  Flow(pipeline).batch_capture(files, width=NETWORK_WIDTH, height=NETWORK_HEIGHT).attach(what=Probe("bboxes", bbox_marker)).freeze().render()
  pipeline.start().wait()


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
