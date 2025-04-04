import argparse
import requests, base64
import time
import os
from typing import List
import cv2
import numpy as np
import mimetypes
import json
import colorsys

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from builder.utils import PayloadBuilder

API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC="shfklsjlfjsljgl"

def create_color_map(num_classes):
    """
    Create a color map for visualizing different classes
    Args:
        num_classes: Number of classes to create colors for
    Returns:
        numpy array of shape (num_classes, 3) with RGB values
    """
    # Corner case: ensure at least 1 class
    if num_classes < 1:
        num_classes = 1

    # Generate evenly spaced colors in HSV space
    hsv_colors = np.zeros((num_classes, 3))
    hsv_colors[:, 0] = np.linspace(0, 1, num_classes, endpoint=False)  # Hue
    hsv_colors[:, 1] = 1.0  # Saturation
    hsv_colors[:, 2] = 1.0  # Value

    # Convert to BGR (what OpenCV uses)
    color_map = np.zeros((num_classes, 3))
    for i in range(num_classes):
        rgb = colorsys.hsv_to_rgb(*hsv_colors[i])
        color_map[i] = (rgb[2] * 255, rgb[1] * 255, rgb[0] * 255)  # BGR format
    return color_map.astype(np.uint8)  # Ensure uint8 type

def draw_label(image, text, x1, y1, bg_color=(255, 255, 0), text_color=(0, 0, 0)):
    """
    Draw a label with background on the image
    Args:
        image: Image to draw on
        text: Text to display
        x1, y1: Top-left corner coordinates where to draw the label
        bg_color: Background color for the label box (default: yellow)
        text_color: Color of the text (default: black)
    Returns:
        Image with label drawn
    """
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Draw label background
    cv2.rectangle(image,
                (x1, y1 - text_height - 4),
                (x1 + text_width + 4, y1),
                bg_color,
                -1)  # -1 fills the rectangle

    # Draw label text
    cv2.putText(image,
               text,
               (x1 + 2, y1 - 2),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               text_color,
               1)

    return image

def convert_bboxes_to_image_size(bboxes, original_shape, target_shape):
    """
    Convert bounding boxes from relative coordinates to target image size
    Args:
        bboxes: List of bounding boxes in relative coordinates
        original_shape: Shape that bboxes are based on (height, width)
        target_shape: Target image shape to convert to (height, width)
    Returns:
        List of bounding boxes in target image coordinates
    """
    if not bboxes:
        return None

    converted_bboxes = []
    for bbox in bboxes:
        x1 = int(bbox[0] / original_shape[1] * target_shape[1])
        y1 = int(bbox[1] / original_shape[0] * target_shape[0])
        x2 = int(bbox[2] / original_shape[1] * target_shape[1])
        y2 = int(bbox[3] / original_shape[0] * target_shape[0])
        converted_bboxes.append([x1, y1, x2, y2])

    return converted_bboxes

def convert_masks_to_image_size(masks_list, original_shape, target_shape):
    """
    Convert masks from relative size to target image size
    Args:
        masks_list: List of masks with label indices
        original_shape: Shape that masks are based on (height, width)
        target_shape: Target image shape to convert to (height, width)
    Returns:
        List of resized masks with original label indices preserved
    """
    if not masks_list:
        return None

    # Validate shapes
    if len(original_shape) != 2 or len(target_shape) != 2:
        raise ValueError("Both original_shape and target_shape must be (height, width)")

    converted_masks = []
    for mask in masks_list:
        try:
            # Convert flattened list back to 2D numpy array
            mask = np.array(mask).reshape(original_shape)

            if mask.max() > 255:
                # If we have more than 255 labels, we need to:
                # 1. Create empty mask using cv2.resize for consistent dimensions
                empty_mask = np.zeros(original_shape, dtype=np.uint8)
                resized_mask = cv2.resize(empty_mask,
                                        (target_shape[1], target_shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)

                # 2. Keep track of unique labels
                unique_labels = np.unique(mask[mask > 0])  # exclude 0 (background)

                # 3. Resize each label's binary mask separately
                for label in unique_labels:
                    binary_mask = (mask == label).astype(np.uint8)
                    resized_binary = cv2.resize(binary_mask,
                                            (target_shape[1], target_shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                    # 4. Restore the original label value
                    resized_mask[resized_binary > 0] = label
            else:
                # If we have fewer than 255 labels, we can resize directly
                resized_mask = cv2.resize(mask.astype(np.uint8),
                                        (target_shape[1], target_shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

            converted_masks.append(resized_mask)
        except Exception as e:
            print(f"Error processing mask: {e}")
            continue

    return converted_masks if converted_masks else None

def overlay_bboxes(image, bboxes, labels=None):
    """
    Create a bounding box overlay on the image with optional labels
    Args:
        image: Original image (numpy array)
        bboxes: List of bounding boxes in image coordinates
        labels: Optional list of string labels
    Returns:
        Image with bounding boxes and labels overlay
    """
    overlay = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Add label if available
        if labels and i < len(labels):
            overlay = draw_label(overlay, labels[i], x1, y1)

    return overlay

def overlay_masks(image, masks, labels=None, alpha=0.5):
    """
    Create a segmentation mask overlay on the image
    Args:
        image: Original image (numpy array)
        masks: List of masks with label indices
        labels: Optional list of string labels
        alpha: Transparency value for the overlay (0-1)
    Returns:
        Image with masks overlay
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image")
    if not masks:
        return image.copy()

    overlay = image.copy()

    # Find the maximum label index to create color map
    max_label = 0
    for mask in masks:
        max_label = max(max_label, mask.max())
    # Create color map for all possible labels (add 1 because 0 is background)
    color_map = create_color_map(max_label + 1)

    for i, mask in enumerate(masks):
        # Create colored mask based on label indices
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for label_idx in range(1, max_label + 1):  # Skip 0 (background)
            mask_for_label = (mask == label_idx)
            if mask_for_label.any():
                colored_mask[mask_for_label] = color_map[label_idx]

        # Create binary mask for alpha blending
        mask_any = (mask > 0)

        # Apply alpha blending only where mask exists
        overlay[mask_any] = cv2.addWeighted(
            overlay[mask_any],
            1.0 - alpha,
            colored_mask[mask_any],
            alpha,
            0
        )

        # Add label if available
        if labels and i < len(labels):
            # Calculate bbox from mask to position the label
            bbox = get_mask_bbox(masks)
            if bbox:
                x1, y1, _, _ = bbox
                overlay = draw_label(overlay, labels[i], x1, y1)

    return overlay

def get_mask_bbox(mask):
    """
    Calculate bounding box coordinates from a binary mask
    Args:
        mask: 2D numpy array binary mask
    Returns:
        tuple (x1, y1, x2, y2) of bbox coordinates, or None if mask is empty
    """
    # Find non-zero points
    y_indices, x_indices = np.nonzero(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    # Get bbox coordinates
    x1 = np.min(x_indices)
    y1 = np.min(y_indices)
    x2 = np.max(x_indices)
    y2 = np.max(y_indices)

    return (x1, y1, x2, y2)

def parse_labels(raw_labels, label_names=None):
    """
    Parse raw label indices into human-readable labels
    Args:
        raw_labels: List of label indices, shape (num_bbox, num_attributes)
        label_names: Optional list of label names to map indices to text
    Returns:
        List of string labels
    """
    if not raw_labels:
        return None

    parsed_labels = []
    for label_data in raw_labels:
        # Convert all attributes to text and join with commas
        label_texts = []
        for attr in label_data:
            # Check if the attribute is a numeric string
            if isinstance(attr, str) and attr.isdigit():
                idx = int(attr)
                if label_names and idx < len(label_names):
                    label_texts.append(label_names[idx])
                else:
                    label_texts.append(str(idx))
            else:
                # If not a numeric string, use the attribute directly
                label_texts.append(str(attr))

        # Join all attributes with commas
        label_text = ", ".join(label_texts)
        parsed_labels.append(label_text)

    return parsed_labels

def visualize_detections(image_path, masks=None, bboxes=None, labels=None, shape=None):
    """
    Visualize detection results combining masks and bounding boxes
    Args:
        image_path: Path to the original image
        masks: Optional list of segmentation masks
        bboxes: Optional list of bounding boxes
        labels: Optional list of string labels
        shape: Required shape that masks/bboxes are based on
    """
    if not shape:
        raise ValueError("Shape parameter is required")

    # Validate input lengths match if provided
    if bboxes and masks and len(bboxes) != len(masks):
        raise ValueError(f"Number of bboxes ({len(bboxes)}) doesn't match number of masks ({len(masks)})")

    if bboxes and labels and len(bboxes) != len(labels):
        raise ValueError(f"Number of bboxes ({len(bboxes)}) doesn't match number of labels ({len(labels)})")

    if masks and labels and not bboxes and len(masks) != len(labels):
        raise ValueError(f"Number of masks ({len(masks)}) doesn't match number of labels ({len(labels)})")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not read image from: {image_path}")

    result = image.copy()
    target_shape = image.shape[:2]  # (height, width)

    # Convert and apply masks if available
    if masks:
        converted_masks = convert_masks_to_image_size(masks, shape, target_shape)
        result = overlay_masks(result, converted_masks,
                             labels if not bboxes else None)

    # Convert and apply bounding boxes if available
    if bboxes:
        converted_bboxes = convert_bboxes_to_image_size(bboxes, shape, target_shape)
        result = overlay_bboxes(result, converted_bboxes, labels)

    cv2.imshow("Inference Results", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def check_empty_2d_list(mask):
    """
    Check if mask from JSON response is [[]] and return None if so
    Args:
        mask: Mask data from JSON response
    Returns:
        None if mask is [[]], otherwise returns mask unchanged
    """
    if isinstance(mask, list) and len(mask) == 1 and isinstance(mask[0], list) and len(mask[0]) == 0:
        return None
    return mask

def dump_response_json(response_data, output_dir="/tmp"):
    """
    Dump response JSON data to a file
    Args:
        response_data: JSON response data to dump
        output_dir: Directory to save the dump file (default: /tmp)
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"infer_response_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(response_data, f)
        print(f"Response JSON dumped to: {filepath}")
    except Exception as e:
        print(f"Error dumping response JSON: {e}")


def main(host , port, model, files, text, dump_response: bool, upload: bool):
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
    payload = { "input": [], "model": f"nvidia/{model}"}
    if upload:
        upload_url = "http://" + host + ":" + port + "/v1/files"
        for file in files:
            multipart_data = MultipartEncoder(
                fields={
                    'file': (os.path.basename(file), open(file, 'rb'), mimetypes.guess_type(file)[0])
                }
            )
            response = requests.post(upload_url, headers={"Content-Type": multipart_data.content_type}, data=multipart_data)
            payload["input"].append(response.json()["data"])
    else:
        mime_types = []
        b64_images = []
        for file in files:
            mime_types.append(mimetypes.guess_type(file)[0])
            with open(file, "rb") as f:
                b64_images.append(base64.b64encode(f.read()).decode())
        payload["input"] = [f"data:{mime_type};base64,{b64_image}" for mime_type, b64_image in zip(mime_types, b64_images)]

    if text:
      payload["text"] = [ t.split(",") for t in text]

    if not upload:
        payload = (PayloadBuilder(files, model)
                    .add_text(text)
                    # .add_more_field("key1", "value1")
                    # .add_more_field("key2", "value2")
                    .build())

    start_time = time.time()
    # This is unofficial client call to quickly visualize results without comprehensive type validation,
    # The validation script uses the official client to validate results against openapi spec
    response = requests.post(invoke_url, headers=headers, json=payload)
    infer_time = time.time() - start_time
    print(response)
    print(infer_time)

    if response.status_code == 200:
        output = response.json()
        if dump_response:
            dump_response_json(output)
            return
        print(f"Usage: num_images= {output['usage']['num_images']}")
        print("Output:")

        # Parse label names from text argument if provided
        label_names = text[0].split(',') if text else None

        for data in output["data"]:
            shape = data["shape"]
            bboxes = data["bboxes"]
            probs = data["probs"]
            raw_labels = data["labels"]
            mask = data["masks"]
            mask=check_empty_2d_list(mask)

            print(f"index = {data['index']}")
            print(f"shape = {shape}")
            print(f"bboxes = {bboxes}")
            print(f"probs = {probs}")
            print(f"labels = {raw_labels}")

            # Parse raw labels into human-readable format
            parsed_labels = parse_labels(raw_labels, label_names) if raw_labels else None

            # Visualize results with whatever data is available
            try:
                visualize_detections(
                    files[data['index']],
                    masks=mask,
                    bboxes=bboxes,
                    labels=parsed_labels,
                    shape=shape
                )
            except ValueError as e:
                print(f"Error visualizing results for image {data['index']}: {e}")
                continue

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
    parser.add_argument("-u", "--upload", action="store_true", help="Upload files to server", default=False)
    parser.add_argument("--dump-response", action="store_true", default=False, help="Enable dumping response JSON to file")

    args = parser.parse_args()
    main(args.host, args.port, args.model, args.file, args.text, args.dump_response, args.upload)
