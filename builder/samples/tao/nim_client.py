import argparse
import requests, base64
import time
import os
from typing import List
import cv2
import mimetypes

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 544

def draw_bboxes(image_path, bboxes, shape):
    """
    Draw bounding boxes on the input image
      Bounding Box Format in nim-tao: [x1, y1, x2, y2]
      - x1, y1: Top-left corner; x2, y2: Bottom-right corner
      - Coordinates are relative to network input dimensions
      - Values range from 0 to network_width/height
      - To convert to image coordinates:
        image_x = bbox_x * (image_width / network_width)
        image_y = bbox_y * (image_height / network_height)

    Example:
      Original image 1920x1080
      Network input 960x544 (shape = [544, 960])
      Model detects a box at [240, 136, 720, 408] (in network coordinates)
      The actual image coordinates would be:
      x1 = 240 / 960 * 1920 = 480
      y1 = 136 / 544 * 1080 = 270
      x2 = 720 / 960 * 1920 = 1440
      y2 = 408 / 544 * 1080 = 810
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    for bbox in bboxes:
        x1 = int(bbox[0] / shape[1] * width)
        y1 = int(bbox[1] / shape[0] * height)
        x2 = int(bbox[2] / shape[1] * width)
        y2 = int(bbox[3] / shape[0] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1)
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def coco_to_network_bbox(coco_bbox, orig_shape, network_shape):
    """Convert COCO format bbox [x,y,w,h] to network relative [x1,y1,x2,y2] format
    Args:
        coco_bbox: List[float] - [x,y,w,h] in absolute image coordinates
        orig_shape: List[int] - [height, width] of original image
        network_shape: List[int] - [height, width] of network input
    Returns:
        List[float] - [x1,y1,x2,y2] normalized to network dimensions
    """
    # First convert COCO [x,y,w,h] to [x1,y1,x2,y2] in image coordinates
    x1 = coco_bbox[0]
    y1 = coco_bbox[1]
    x2 = coco_bbox[0] + coco_bbox[2]
    y2 = coco_bbox[1] + coco_bbox[3]

    # Then normalize to network dimensions
    x1_norm = x1 / orig_shape[1] * network_shape[1]
    y1_norm = y1 / orig_shape[0] * network_shape[0]
    x2_norm = x2 / orig_shape[1] * network_shape[1]
    y2_norm = y2 / orig_shape[0] * network_shape[0]

    return [x1_norm, y1_norm, x2_norm, y2_norm]

def network_to_coco_bbox(network_bbox, orig_shape, network_shape):
    """Convert network relative [x1,y1,x2,y2] to COCO format [x,y,w,h]
    Args:
        network_bbox: List[float] - [x1,y1,x2,y2] normalized to network dimensions
        orig_shape: List[int] - [height, width] of original image
        network_shape: List[int] - [height, width] of network input
    Returns:
        List[float] - [x,y,w,h] in absolute image coordinates
    """
    # First denormalize to image coordinates
    x1 = network_bbox[0] / network_shape[1] * orig_shape[1]
    y1 = network_bbox[1] / network_shape[0] * orig_shape[0]
    x2 = network_bbox[2] / network_shape[1] * orig_shape[1]
    y2 = network_bbox[3] / network_shape[0] * orig_shape[0]

    # Then convert [x1,y1,x2,y2] to COCO format [x,y,w,h]
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1

    return [x, y, w, h]

def main(host , port, model, files, upload):
    if not files:
        print("Need the file path for inference")
        return

    invoke_url = "http://" + host + ":" + port + "/v1/inference"
    headers = {
        "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
        "Accept": "application/json",
        # "NVCF-ASSET-DIR": "some-temp-dir",
        # "NVCF-FUNCTION-ASSET-IDS": "udjflsjo-jfoisjof-lsdfjofdj"
    }
    if upload:
        upload_url = "http://" + host + ":" + port + "/v1/files"
        # Create multipart form-data payload for file upload
        for file in files:
            with open(file, 'rb') as f:
                files = {"file": (file, f, 'application/octet-stream')}
                # Send multipart request
                response = requests.post(upload_url, headers=headers, files=files)
                print(response)
        payload = {}
        return
    else:
        mime_types = []
        b64_images = []
        for file in files:
            mime_types.append(mimetypes.guess_type(file)[0])
            with open(file, "rb") as f:
                b64_images.append(base64.b64encode(f.read()).decode())

        # assert len(image_b64) < 180_000, \
        #   "To upload larger images, use the assets API (see docs)"



        payload = {
            "input": [f"data:{mime_type};base64,{b64_image}" for mime_type, b64_image in zip(mime_types, b64_images)],
            "model": f"nvidia/{model}"
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
            shape = data["shape"]
            bboxes = data["bboxes"]
            probs = data["probs"]
            labels = data["labels"]
            print(f"index = {data['index']}")
            print(f"shape = {shape}")
            print(f"bboxes = {bboxes}")
            print(f"probs = {probs}")
            print(f"labels = {labels}")
            bboxes_list.append(bboxes)

        for file, bboxes in zip(files, bboxes_list):
            draw_bboxes(file, bboxes, shape)

    elif response.status_code == 422:
        print("Unable to process request: 422. Check the payload")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str,
                      help= "Server IP Address", default="127.0.0.1")
    parser.add_argument("--port", type=str,
                      help="Server port", default="8000")
    parser.add_argument("--model", type=str, help="Model name", default="nvdino-v2")
    parser.add_argument("--file", type=str, help="File to send for inference", nargs='*', default=None)
    parser.add_argument("-u", "--upload", action="store_true", help="Upload the file to server")

    args = parser.parse_args()
    main(args.host, args.port, args.model, args.file, args.upload)
