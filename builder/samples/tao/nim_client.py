import argparse
import requests, base64
import time
import os
from typing import List
import cv2

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"

def overlay_segmentation_mask(image_path, mask, alpha=0.5):
    """
    Overlay segmentation mask on the original image
    Args:
        image_path: Path to the original image
        mask: Segmentation mask array
        alpha: Transparency value for the overlay (0-1)
    Returns:
        Overlaid image
    """
    # Read original image
    image = cv2.imread(image_path)

    # Resize mask to match image dimensions
    height, width = image.shape[:2]
    mask = cv2.resize(mask, (width, height))

    # Create colored mask (you can modify colors as needed)
    colored_mask = cv2.applyColorMap(mask.astype('uint8'), cv2.COLORMAP_JET)

    # Overlay mask on image
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)

    return overlay


def draw_bboxes(image_path, bboxes, shape):
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

def main(host , port, model, files, text):
    if not files:
        print("Need the file path for inference")
        return

    invoke_url = "http://" + host + ":" + port + "/v1/inference"

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
      "model": f"nvidia/{model}"
    }

    if text:
      payload["text"] = [ t.split(",") for t in text]

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
            mask = data["mask"]
            print(f"index = {data['index']}")
            print(f"shape = {shape}")
            print(f"bboxes = {bboxes}")
            print(f"probs = {probs}")
            print(f"labels = {labels}")
            print(f"mask = {mask}")
            bboxes_list.append(bboxes)
            if bboxes:
                draw_bboxes(file, bboxes, shape)
            if mask:
                overlay_segmentation_mask(file, mask)
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
    parser.add_argument("--text", type=str, help="Extra text to send for inference", nargs='*', default=None)

    args = parser.parse_args()
    main(args.host, args.port, args.model, args.file, args.text)
