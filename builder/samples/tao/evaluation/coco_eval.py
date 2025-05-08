import argparse
import json
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                   level=logging.INFO)
logger = logging.getLogger(__name__)

class COCOEvaluator:
    def __init__(self, gt_file):
        """Initialize COCO evaluator with ground truth file.
        
        Args:
            gt_file (str): Path to COCO format ground truth JSON file
        """
        self.coco_gt = COCO(gt_file)
        
    def evaluate(self, predictions, ann_type='bbox'):
        """Run COCO evaluation.
        
        Args:
            predictions (list): List of detections in COCO format. Each detection is a dict:
                {
                    "image_id": int,        # Image ID
                    "category_id": int,     # Category ID
                    "bbox": [x,y,w,h],      # Bounding box in [x,y,width,height] format
                    "score": float          # Detection confidence score
                }
            ann_type (str): Annotation type ('bbox' or 'segm')
            
        Returns:
            dict: Evaluation metrics including AP, AP50, AP75, etc.
        """
        # Load results into COCO API
        coco_dt = self.coco_gt.loadRes(predictions)
        
        # Create evaluator
        coco_eval = COCOeval(self.coco_gt, coco_dt, ann_type)
        
        # # Get unique image IDs from predictions
        image_ids = list(set(pred['image_id'] for pred in predictions))
        coco_eval.params.imgIds = sorted(image_ids)
        # coco_eval.params.imgIds = sorted(self.coco_gt.getImgIds())
        coco_eval.params.useCats = 0

        logger.info(f"Running evaluation with {len(coco_eval.params.imgIds)} images")
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Return metrics
        return {
            'AP': coco_eval.stats[0],    # AP @[0.5:0.95]
            'AP50': coco_eval.stats[1],  # AP @0.5
            'AP75': coco_eval.stats[2],  # AP @0.75
            'APs': coco_eval.stats[3],   # AP small
            'APm': coco_eval.stats[4],   # AP medium
            'APl': coco_eval.stats[5],   # AP large
            'AR1': coco_eval.stats[6],   # AR @1
            'AR10': coco_eval.stats[7],  # AR @10
            'AR100': coco_eval.stats[8], # AR @100
            'ARs': coco_eval.stats[9],   # AR small
            'ARm': coco_eval.stats[10],  # AR medium
            'ARl': coco_eval.stats[11]   # AR large
        }

def parse_args():
    parser = argparse.ArgumentParser(description='COCO Evaluation Script')
    parser.add_argument('--gt_file', type=str, required=True,
                      help='Path to COCO format ground truth JSON file')
    parser.add_argument('--dt_file', type=str, required=True,
                      help='Path to COCO format detection results JSON file')
    parser.add_argument('--ann_type', type=str, default='bbox',
                      choices=['bbox', 'segm'],
                      help='Annotation type (bbox or segm)')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Path to save evaluation results (optional)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load predictions
    logger.info(f"Loading predictions from {args.dt_file}")
    with open(args.dt_file, 'r') as f:
        predictions = json.load(f)
    
    # Initialize evaluator
    logger.info(f"Initializing evaluator with ground truth from {args.gt_file}")
    evaluator = COCOEvaluator(args.gt_file)
    
    # Run evaluation
    logger.info(f"Running {args.ann_type} evaluation")
    metrics = evaluator.evaluate(predictions, ann_type=args.ann_type)
    
    # Print results
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Save results if output file specified
    if args.output_file:
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
