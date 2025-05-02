import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import sys

# Add the project root to the Python path to import nim_client
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from nim_client import main as nim_client_main, convert_bboxes_to_image_size

def single_inference_to_coco_predictions(inference_response: Dict, image_id: int, target_shape: tuple) -> List[Dict]:
    """Convert a single inference response to COCO prediction format.
    
    Args:
        inference_response: Response from the inference server
        image_id: Image ID to use in the COCO predictions
        target_shape: Actual image shape (height, width) to convert bbox coordinates to

    Returns:
        List of predictions in COCO format
    """
    predictions = []
    
    # Process each detection in the response
    for data in inference_response["data"]:
        original_shape = tuple(data["shape"])  # Shape that bboxes are relative to
        bboxes = data["bboxes"]
        scores = data["probs"]
        labels = data["labels"]
        
        # Convert bboxes to target image size
        converted_bboxes = convert_bboxes_to_image_size(bboxes, original_shape, target_shape)
        if not converted_bboxes:
            continue

        # Convert each detection to COCO format
        for bbox, score, label in zip(converted_bboxes, scores, labels):
            # Convert [x1, y1, x2, y2] to [x, y, width, height]
            x = bbox[0]
            y = bbox[1]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            prediction = {
                "image_id": image_id,
                "bbox": [x, y, width, height],
                "score": score,
                "category_id": int(label[0])  # Assuming label is a list with one element
            }
            predictions.append(prediction)
    
    return predictions

def run_evaluation(config_path: str, image_base_dir: str, host: str, port: str, output_path: str):
    """Run evaluation on a set of images and save COCO predictions.
    
    Args:
        config_path: Path to the validation config file
        image_base_dir: Base directory containing the images
        host: Inference server host
        port: Inference server port
        output_path: Path to save the COCO predictions
    """
    # Load validation config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    all_predictions = []
    
    # Process each image in the config
    for image_entry in config["images"]:
        image_name = image_entry["file_name"]
        image_id = image_entry["id"]
        target_shape = (image_entry["height"], image_entry["width"])  # Get target shape from config
        
        # Construct full image path
        image_path = os.path.join(image_base_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        print(f"Processing image: {image_path}")
        
        # Call nim_client with the image
        # Capture the response by modifying nim_client to return the response
        response = nim_client_main(
            host=host,
            port=port,
            model="nvdino-v2",  # You might want to make this configurable
            files=[image_path],
            text=None,
            dump_response=False,
            upload=False,
            return_response=True  # New parameter to return response instead of printing
        )
        
        print(f"Response:\n{response}")
        if response is None:
            print(f"Warning: Failed to get response for {image_path}")
            continue
        
        # Convert to COCO format
        coco_predictions = single_inference_to_coco_predictions(response, image_id, target_shape)
        all_predictions.extend(coco_predictions)
    
    # Save all predictions to file
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"Saved predictions to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation on a set of images')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to validation config file')
    parser.add_argument('--image-dir', type=str, required=True,
                      help='Base directory containing the images')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Inference server host')
    parser.add_argument('--port', type=str, default='8800',
                      help='Inference server port')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save COCO predictions')
    return parser.parse_args()

def main():
    args = parse_args()
    run_evaluation(
        config_path=args.config,
        image_base_dir=args.image_dir,
        host=args.host,
        port=args.port,
        output_path=args.output
    )

if __name__ == '__main__':
    main()