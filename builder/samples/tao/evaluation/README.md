## Helper to get a subset from the complete set
```python
python3 create_debug_subset.py /media/scratch.metropolis3/yuw/datasets/coco/annotations/instances_val2017.json /tmp/val2017_vehicle.50.json --supercategory vehicle --num_images 50
```

## Eval usage:
```python
python3 combined_eval.py --val-json=/tmp/val2017_vehicle.50.json --image-dir=/media/scratch.metropolis3/yuw/datasets/coco/val2017 --output=/tmp/tao/pred_val2017_vehicle.50.json --host 10.111.53.46 --dump-vis-path=/tmp/tao/vis_val2017_vehicle.50
```

## COCO Prediction schema
```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "examples": [
      [
        {
            "image_id": 1,
            "bbox": [366.95068359, 386.24633789, 17.97363281, 17.46826172],
            "score": 0.9428234,
            "category_id": 2
        },
        {
            "image_id": 1,
            "bbox": [548.27636719, 32.32040405, 5.94726562, 5.95733643],
            "score": 0.93891287,
            "category_id": 2
        }
      ]
    ],
    "items": {
      "type": "object",
      "properties": {
        "image_id": {
          "type": "integer",
          "description": "Image identifier from source_id"
        },
        "bbox": {
          "type": "array",
          "items": {
            "type": "number"
          },
          "minItems": 4,
          "maxItems": 4,
          "description": "Bounding box coordinates [x, y, w, h] from detection_boxes"
        },
        "score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Detection confidence score from detection_scores"
        },
        "category_id": {
          "type": "integer",
          "description": "Category identifier from detection_classes"
        }
      },
      "required": ["image_id", "bbox", "score", "category_id"]
    }
  }
```