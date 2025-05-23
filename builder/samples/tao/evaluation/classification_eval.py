import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---- Logger setup ----
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# ----------------------

# Add the project root to the Python path to import nim_client
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from nim_client import main as nim_client_main

class DataCollector:
    """Class responsible for collecting ground truth and predictions."""
    
    def __init__(self, root_dir: str, test_prefix: str, max_images_per_class: int = None, total_images: int = None):
        """Initialize data collector.
        
        Args:
            root_dir: Root directory of the dataset
            test_prefix: Prefix path to test directory
            max_images_per_class: Maximum number of images to take per class (optional)
            total_images: Total number of images to evaluate (optional)
            
        Example directory structure:
        root_dir/
        ├── imagenet/
        │   └── val/
        │       ├── n01440764/
        │       │   ├── ILSVRC2012_val_00000001.JPEG
        │       │   └── ILSVRC2012_val_00000002.JPEG
        │       └── n01443537/
        │           ├── ILSVRC2012_val_00000003.JPEG
        │           └── ILSVRC2012_val_00000004.JPEG
        """
        self.root_dir = root_dir
        self.test_dir = os.path.join(root_dir, test_prefix)
        self.max_images_per_class = max_images_per_class
        self.total_images = total_images
        
    def get_image_label_pairs(self) -> List[Tuple[str, str]]:
        """Get list of (image_path, ground_truth_label) pairs.
        
        Example return:
        [
            ("/path/to/val/n01440764/ILSVRC2012_val_00000001.JPEG", "n01440764"),
            ("/path/to/val/n01440764/ILSVRC2012_val_00000002.JPEG", "n01440764"),
            ...
        ]
        """
        image_label_pairs = []
        
        # Get list of class directories
        class_dirs = [d for d in os.listdir(self.test_dir) 
                     if os.path.isdir(os.path.join(self.test_dir, d))]
        
        # Use tqdm for class directories
        for class_dir in tqdm(class_dirs, desc="Loading class directories"):
            class_path = os.path.join(self.test_dir, class_dir)
            
            # Get all images in this class directory
            img_names = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Apply per-class limit if specified
            if self.max_images_per_class:
                img_names = img_names[:self.max_images_per_class]
            
            # Add image-label pairs
            for img_name in img_names:
                img_path = os.path.join(class_path, img_name)
                image_label_pairs.append((img_path, class_dir))
        
        # Apply total images limit if specified
        if self.total_images and self.total_images < len(image_label_pairs):
            # Randomly sample if total_images is specified
            import random
            random.shuffle(image_label_pairs)
            image_label_pairs = image_label_pairs[:self.total_images]
            
        logger.info(f"Loaded {len(image_label_pairs)} images from {len(class_dirs)} classes")
        return image_label_pairs

class PredictionCollector:
    """Class responsible for collecting model predictions."""
    
    def __init__(self, host: str, port: str, model_name: str = "nvdino-v2", 
                 csv_path: Optional[str] = None, dump_csv: Optional[str] = None):
        """Initialize prediction collector.
        
        Args:
            host: Inference server host
            port: Inference server port
            model_name: Model name for inference
            csv_path: Path to CSV file containing predictions (optional)
            dump_csv: Path to dump predictions in CSV format (optional)
        """
        self.host = host
        self.port = port
        self.model_name = model_name
        self.csv_path = csv_path
        self.dump_csv = dump_csv
        self.predictions_df = None
        
        # Load predictions from CSV if provided
        if csv_path:
            # Read CSV without headers, columns are: image_path, pred_label, confidence
            self.predictions_df = pd.read_csv(csv_path, header=None)
            logger.info(f"Loaded {len(self.predictions_df)} predictions from {csv_path}")
        
    def get_prediction(self, image_path: str) -> str:
        """Get prediction for a single image.
        
        If csv_path was provided during initialization, predictions are loaded from CSV.
        Otherwise, predictions are obtained from the inference server.
        
        Example inference_response:
        {
            "data": [{
                "index": 0,
                "shape": [224, 224],
                "bboxes": [[0, 0, 224, 224]],
                "probs": [0.95],
                "labels": [["n01440764"]],
                "masks": [],
                "timestamp": 151950801291
            }],
            "model": "nvidia/nvdinov2"
        }
        
        Returns:
            str: Predicted class label (e.g., "n01440764")
        """
        # If using CSV predictions
        if self.predictions_df is not None:
            # Column 0 is image_path, column 1 is pred_label
            pred_row = self.predictions_df[self.predictions_df[0] == image_path]
            if len(pred_row) == 0:
                logger.warning(f"No prediction found in CSV for {image_path}")
                return None
            return pred_row.iloc[0][1]  # Get pred_label from column 1
        
        # Otherwise, get prediction from inference server
        response = nim_client_main(
            host=self.host,
            port=self.port,
            model=self.model_name,
            files=[image_path],
            text=None,
            dump_response=False,
            upload=False,
            return_response=True
        )
        
        if not response or "data" not in response:
            logger.warning(f"Failed to get prediction for {image_path}")
            return None
            
        # Get the first (and only) prediction
        data = response["data"][0]
        if not data["labels"]:
            logger.warning(f"No labels in prediction for {image_path}")
            return None
            
        # Return the first label (we expect only one)
        return data["labels"][0][0]
    
    def dump_predictions(self, image_paths: List[str], predictions: List[str]):
        """Dump predictions to CSV file.
        
        Args:
            image_paths: List of image paths
            predictions: List of predicted labels
        """
        if not self.dump_csv:
            return
            
        # Create DataFrame without column names, matching reference code format
        df = pd.DataFrame(zip(image_paths, predictions, [0.0] * len(predictions)))
        
        # Append to CSV file without headers
        df.to_csv(self.dump_csv, header=False, index=False, mode='a')
        logger.info(f"Dumped {len(predictions)} predictions to {self.dump_csv}")

class MetricsCalculator:
    """Class responsible for calculating classification metrics."""
    
    def __init__(self, all_labels: List[str]):
        """Initialize metrics calculator.
        
        Args:
            all_labels: List of all possible class labels
        """
        self.all_labels = all_labels
        
    def compute_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """Compute classification metrics.
        
        Example inputs:
        y_true = ["n01440764", "n01440764", "n01443537", "n01443537"]
        y_pred = ["n01440764", "n01443537", "n01443537", "n01443537"]
        
        Returns:
            Dict containing:
            - accuracy: Overall accuracy
            - precision: Per-class precision
            - recall: Per-class recall
            - f1: Per-class F1 score
            - confusion_matrix: Confusion matrix
            - class_metrics: Dict mapping class names to their metrics
            - overall_metrics: Dict containing macro/micro averages
            
        Note on Macro vs Micro averages:
        - Macro: Treats all classes equally, regardless of their frequency
          Example: If we have 3 classes with metrics:
          Class A (100 samples): precision=0.9, recall=0.8
          Class B (10 samples):  precision=0.5, recall=0.4
          Class C (1 sample):   precision=0.3, recall=0.2
          Macro average = (0.9 + 0.5 + 0.3) / 3 = 0.57
          This is useful when all classes are equally important.
        
        - Micro: Weights metrics by class frequency
          Example: Using the same classes above:
          Class A: 100 samples contribute 90% to final score
          Class B: 10 samples contribute 9% to final score
          Class C: 1 sample contributes 1% to final score
          Micro average ≈ 0.9 (dominated by Class A)
          This is useful when class imbalance exists and frequent classes are more important.
        """
        # Convert string labels to indices for sklearn metrics
        label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        y_true_idx = [label_to_idx[label] for label in y_true]
        y_pred_idx = [label_to_idx[label] for label in y_pred]
        
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy
        metrics["accuracy"] = accuracy_score(y_true_idx, y_pred_idx)
        
        # Per-class metrics with zero_division=0 to handle warnings
        # zero_division=0 means: if a class has no predictions, its precision=0
        # if a class has no ground truth, its recall=0
        metrics["precision"] = precision_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
        metrics["recall"] = recall_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
        metrics["f1"] = f1_score(y_true_idx, y_pred_idx, average=None, zero_division=0)
        
        # Overall metrics (macro and micro averages)
        metrics["overall_metrics"] = {
            "macro_precision": precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
            "macro_recall": recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
            "macro_f1": f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0),
            "micro_precision": precision_score(y_true_idx, y_pred_idx, average='micro', zero_division=0),
            "micro_recall": recall_score(y_true_idx, y_pred_idx, average='micro', zero_division=0),
            "micro_f1": f1_score(y_true_idx, y_pred_idx, average='micro', zero_division=0)
        }
        
        # Create confusion matrix with all classes
        # Initialize with zeros to include classes with no predictions
        conf_matrix = np.zeros((len(self.all_labels), len(self.all_labels)), dtype=int)
        for true_idx, pred_idx in zip(y_true_idx, y_pred_idx):
            conf_matrix[true_idx, pred_idx] += 1
        metrics["confusion_matrix"] = conf_matrix
        
        # Create class metrics dictionary
        metrics["class_metrics"] = {}
        for idx, label in enumerate(self.all_labels):
            metrics["class_metrics"][label] = {
                "precision": metrics["precision"][idx],
                "recall": metrics["recall"][idx],
                "f1": metrics["f1"][idx],
                "true_count": np.sum(conf_matrix[idx, :]),  # Number of true samples
                "pred_count": np.sum(conf_matrix[:, idx])   # Number of predicted samples
            }
        
        return metrics

class ClassificationEvaluator:
    """Main class for classification evaluation."""
    
    def __init__(self, host: str, port: str, model_name: str = "nvdino-v2", config_path: str = None,
                 max_images_per_class: int = None, total_images: int = None,
                 csv_path: Optional[str] = None, dump_csv: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            host: Inference server host
            port: Inference server port
            model_name: Model name for inference
            config_path: Path to experiment config YAML file
            max_images_per_class: Maximum number of images to take per class (optional)
            total_images: Total number of images to evaluate (optional)
            csv_path: Path to CSV file containing predictions (optional)
            dump_csv: Path to dump predictions in CSV format (optional)
            
        Example config YAML structure:
        dataset:
          root_dir: /path/to/dataset
          test:
            data_prefix: imagenet/val
        """
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Extract dataset info
            dataset_config = self.config['dataset']
            self.root_dir = dataset_config['root_dir']
            self.test_prefix = dataset_config['test']['data_prefix']
        
        # Initialize components
        self.data_collector = DataCollector(
            self.root_dir, 
            self.test_prefix,
            max_images_per_class=max_images_per_class,
            total_images=total_images
        )
        self.pred_collector = PredictionCollector(
            host=host,
            port=port,
            model_name=model_name,
            csv_path=csv_path,
            dump_csv=dump_csv
        )
        
    def evaluate(self) -> Dict:
        """Run evaluation.
        
        Returns:
            Dict containing evaluation metrics
            
        Note on handling missing predictions:
        When a prediction fails (no label returned), we:
        1. Skip that image in metrics calculation
        2. Track the ground truth class in failed_predictions set
        3. Report these classes at the end
        This helps identify classes that the model struggles with.
        """
        # Get all image-label pairs
        image_label_pairs = self.data_collector.get_image_label_pairs()
        logger.info(f"Found {len(image_label_pairs)} images to evaluate")
        
        # Collect predictions
        y_true = []
        y_pred = []
        all_labels = set()
        failed_predictions = set()  # Track classes that had failed predictions
        image_paths = []
        predictions = []
        
        for img_path, gt_label in tqdm(image_label_pairs, desc="Processing images"):
            # Get prediction
            pred_label = self.pred_collector.get_prediction(img_path)
            if pred_label is None:
                # If prediction fails, track the ground truth class
                failed_predictions.add(gt_label)
                continue
                
            # Store results
            y_true.append(gt_label)
            y_pred.append(pred_label)
            all_labels.add(gt_label)
            all_labels.add(pred_label)
            
            # Store for CSV dumping
            image_paths.append(img_path)
            predictions.append(pred_label)
        
        # Dump predictions if requested
        if image_paths:
            self.pred_collector.dump_predictions(image_paths, predictions)
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator(sorted(list(all_labels)))
        metrics = metrics_calculator.compute_metrics(y_true, y_pred)
        
        # Print results
        logger.info("\nEvaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Print overall metrics
        overall = metrics["overall_metrics"]
        logger.info("\nOverall Metrics:")
        logger.info(f"Macro Precision: {overall['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {overall['macro_recall']:.4f}")
        logger.info(f"Macro F1: {overall['macro_f1']:.4f}")
        logger.info(f"Micro Precision: {overall['micro_precision']:.4f}")
        logger.info(f"Micro Recall: {overall['micro_recall']:.4f}")
        logger.info(f"Micro F1: {overall['micro_f1']:.4f}")
        
        # Print classes with no predictions
        if failed_predictions:
            logger.info(f"\nClasses with no predictions: {len(failed_predictions)}")
            for label in sorted(failed_predictions):
                logger.info(f"  {label}")
        
        # Print top and bottom performing classes
        class_metrics = metrics["class_metrics"]
        sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
        
        logger.info("\nTop 5 performing classes:")
        for class_name, class_metrics in sorted_classes[:5]:
            logger.info(f"\n{class_name}:")
            logger.info(f"  Precision: {class_metrics['precision']:.4f}")
            logger.info(f"  Recall: {class_metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {class_metrics['f1']:.4f}")
            logger.info(f"  True samples: {class_metrics['true_count']}")
            logger.info(f"  Predicted samples: {class_metrics['pred_count']}")
            
        logger.info("\nBottom 5 performing classes:")
        for class_name, class_metrics in sorted_classes[-5:]:
            logger.info(f"\n{class_name}:")
            logger.info(f"  Precision: {class_metrics['precision']:.4f}")
            logger.info(f"  Recall: {class_metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {class_metrics['f1']:.4f}")
            logger.info(f"  True samples: {class_metrics['true_count']}")
            logger.info(f"  Predicted samples: {class_metrics['pred_count']}")
        
        # Print confusion matrix dimensions
        conf_matrix = metrics['confusion_matrix']
        logger.info(f"\nConfusion Matrix Shape: {conf_matrix.shape}")
        
        return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Run classification evaluation')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to experiment config YAML file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Inference server host')
    parser.add_argument('--port', type=str, default='8800',
                      help='Inference server port')
    parser.add_argument('--max-images-per-class', type=int,
                      help='Maximum number of images to evaluate per class')
    parser.add_argument('--total-images', type=int,
                      help='Total number of images to evaluate (randomly sampled)')
    parser.add_argument('--csv-path', type=str,
                      help='Path to CSV file containing predictions')
    parser.add_argument('--dump-csv', type=str,
                      help='Path to dump predictions in CSV format')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize evaluator
    evaluator = ClassificationEvaluator(
        host=args.host,
        port=args.port,
        config_path=args.config,
        max_images_per_class=args.max_images_per_class,
        total_images=args.total_images,
        csv_path=args.csv_path,
        dump_csv=args.dump_csv
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()

if __name__ == '__main__':
    main() 