#!/usr/bin/env python3
import argparse
import json
import numpy as np
from typing import Dict, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def single_inference_to_coco_predictions(inference_response: Dict, image_id: int = 0) -> List[Dict]:
    """Convert a single image inference response to COCO prediction format.
    
    Args:
        inference_response: Dict, inference response for a single image in format:
            {
                "data": [{
                    "shape": [height, width],
                    "bboxes": [[x1, y1, x2, y2], ...],
                    "probs": [score1, score2, ...],
                    "labels": [["class_id"], ["class_id"], ...],
                    "masks": [[mask1_flattened], [mask2_flattened], ...],  # Optional
                    "timestamp": int
                }],
                "model": str,
                "usage": {"num_images": 1}
            }
        image_id: int, identifier for this image
    
    Returns:
        List of dictionaries in COCO format:
        [
            {
                'image_id': int,
                'bbox': [x, y, w, h],
                'score': float,
                'category_id': int,
                'segmentation': RLE  # Optional, if masks provided
            },
            ... (one dict per detection)
        ]
    """
    predictions = []
    
    # Extract detection data from response
    detections = inference_response['data'][0]  # Single image data
    boxes = np.array(detections['bboxes'])  # [[x1, y1, x2, y2], ...]
    scores = detections['probs']
    labels = detections['labels']  # [["3"], ["3"], ...]
    masks = detections['masks']  # [[mask1_flattened], [mask2_flattened], ...]
    image_height, image_width = detections['shape']
    
    has_masks = len(masks) > 0
    
    # Process each detection
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Convert box from [x1, y1, x2, y2] to COCO format [x, y, w, h]
        x, y = box[0], box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        prediction = {
            'image_id': image_id,
            'bbox': [float(x), float(y), float(w), float(h)],
            'score': float(score),
            'category_id': int(label[0])
        }
        
        # Handle segmentation if masks are present
        if has_masks:
            from pycocotools import mask as maskUtils  # Need this for RLE encoding
            # Reshape flattened mask to 2D
            mask_flat = np.array(masks[i])
            mask_2d = mask_flat.reshape(image_height, image_width)
            
            # Convert to uint8 and then to fortranarray for RLE encoder
            mask_2d = mask_2d.astype(np.uint8)
            mask_fortran = np.asfortranarray(mask_2d)
            
            # Encode mask using RLE
            rle = maskUtils.encode(mask_fortran)
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            
            prediction['segmentation'] = rle
        
        predictions.append(prediction)
    
    return predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Convert inference response to COCO format')
    parser.add_argument('--input', type=str, help='Input JSON string or file path')
    parser.add_argument('--image-id', type=int, default=0, help='Image ID for the predictions')
    parser.add_argument('--output', type=str, default='coco_predictions.json', 
                       help='Output JSON file path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load input
    try:
        # First try to parse as JSON string
        inference_response = json.loads(args.input)
    except json.JSONDecodeError:
        try:
            # If that fails, try to load as file
            with open(args.input, 'r') as f:
                inference_response = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load input: {e}")
            return

    # Convert to COCO format
    try:
        coco_predictions = single_inference_to_coco_predictions(
            inference_response, 
            image_id=args.image_id
        )
        
        # Print summary
        logger.info(f"Converted {len(coco_predictions)} detections to COCO format")
        has_masks = 'segmentation' in coco_predictions[0] if coco_predictions else False
        logger.info(f"Predictions include segmentation masks: {has_masks}")
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump(coco_predictions, f, indent=2)
        logger.info(f"Saved COCO predictions to {args.output}")
        
        # Print first prediction as example
        if coco_predictions:
            logger.info("Example prediction:")
            logger.info(json.dumps(coco_predictions[0], indent=2))
            
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return

if __name__ == "__main__":
    main()
