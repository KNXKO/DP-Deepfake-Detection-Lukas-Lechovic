import os
import sys

# Add training directory to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'training')
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yaml
import pickle
import json
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

import argparse


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detector Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test Xception model
  python evaluate_detector.py --detector_path ../training/config/detector/xception.yaml \\
                              --weights_path ../training/weights/xception_best.pth \\
                              --test_dataset MyDataset_full

  # Resume interrupted test
  python evaluate_detector.py --detector_path ../training/config/detector/meso4.yaml \\
                              --weights_path ../training/weights/meso4_best.pth \\
                              --resume
        """
    )
    parser.add_argument(
        '--detector_path',
        type=str,
        required=True,
        help='Path to detector YAML configuration file'
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default='MyDataset_full',
        help='Name of test dataset (default: MyDataset_full)'
    )
    parser.add_argument(
        '--weights_path',
        type=str,
        required=True,
        help='Path to model weights file (.pth)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./evaluation_results',
        help='Directory for checkpoints and results (default: ./evaluation_results)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    return parser.parse_args()


# Device detection (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
#  CHECKPOINT MANAGER
# ==============================================================================

class CheckpointManager:
    """
    Checkpoint manager for long-running tests.

    Enables saving and loading intermediate results,
    preventing data loss on test interruption.
    """

    def __init__(self, checkpoint_dir: str, test_name: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            test_name: Test name (used in file names)
        """
        self.checkpoint_dir = checkpoint_dir
        self.test_name = test_name
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.checkpoint_file = os.path.join(
            checkpoint_dir,
            f'{test_name}_{timestamp}_checkpoint.pkl'
        )
        self.final_results_file = os.path.join(
            checkpoint_dir,
            f'{test_name}_{timestamp}_results.pkl'
        )

    def save_checkpoint(self, predictions: list, labels: list,
                       image_names: list, batch_idx: int) -> None:
        """Save current test state."""
        checkpoint = {
            'predictions': predictions,
            'labels': labels,
            'image_names': image_names,
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"\n[CHECKPOINT] Saved at batch {batch_idx}")

    def load_checkpoint(self) -> dict:
        """Load last checkpoint."""
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.test_name) and f.endswith('_checkpoint.pkl')
        ]

        if not checkpoints:
            return None

        latest = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest)

        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        print(f"\n[RESUME] Loaded checkpoint from batch {checkpoint['batch_idx']}")
        print(f"[RESUME] Timestamp: {checkpoint['timestamp']}")
        return checkpoint

    def save_final_results(self, results: dict) -> None:
        """Save final test results."""
        with open(self.final_results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n[RESULTS] Saved to {self.final_results_file}")


# ==============================================================================
#  EVALUATION LOGIC
# ==============================================================================

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                   checkpoint_mgr: CheckpointManager, resume: bool = False) -> tuple:
    """
    Evaluate model on entire dataset with checkpoint support.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with test data
        checkpoint_mgr: Checkpoint manager
        resume: Resume from last checkpoint

    Returns:
        tuple: (predictions, actual labels)
    """
    model.eval()

    prediction_lists = []
    label_lists = []
    image_name_lists = []
    start_batch = 0

    # Load checkpoint if requested
    if resume:
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            prediction_lists = checkpoint['predictions']
            label_lists = checkpoint['labels']
            image_name_lists = checkpoint['image_names']
            start_batch = checkpoint['batch_idx'] + 1
            print(f"[RESUME] Continuing from batch {start_batch}/{len(data_loader)}")

    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(data_loader, initial=start_batch,
                                            total=len(data_loader),
                                            desc="Evaluation")):
            if i < start_batch:
                continue

            # Get data
            data, label = data_dict['image'], data_dict['label']
            label = torch.where(label != 0, 1, 0)

            # Move to device (GPU/CPU)
            data, label = data.to(DEVICE), label.to(DEVICE)
            data_dict['image'], data_dict['label'] = data, label

            if data_dict.get('mask') is not None:
                data_dict['mask'] = data_dict['mask'].to(DEVICE)
            if data_dict.get('landmark') is not None:
                data_dict['landmark'] = data_dict['landmark'].to(DEVICE)

            # Forward pass
            predictions = model(data_dict, inference=True)

            # Collect results
            label_lists.extend(label.cpu().numpy().tolist())
            prediction_lists.extend(predictions['prob'].cpu().numpy().tolist())

            # Checkpoint every 200 batches
            if (i + 1) % 200 == 0:
                checkpoint_mgr.save_checkpoint(
                    prediction_lists, label_lists, [], i
                )

    return np.array(prediction_lists), np.array(label_lists)


# ==============================================================================
#  METRICS COMPUTATION
# ==============================================================================

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                    output_dir: str) -> dict:
    """
    Compute and save all evaluation metrics.

    Generates:
      - metrics.json: All numeric metrics
      - roc_curve.png: ROC curve with AUC
      - precision_recall_curve.png: Precision-Recall curve
      - confusion_matrix.png: Confusion matrix visualization

    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: Actual labels (0 or 1)
        output_dir: Directory for saving outputs

    Returns:
        dict: Dictionary with all metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Binary predictions (threshold 0.5)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Precision-Recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred)

    # Build metrics dictionary
    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc),
        'eer': float(eer),
        'specificity': float(specificity),
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        },
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true)),
        'negative_samples': int(len(y_true) - np.sum(y_true))
    }

    # Save metrics to JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Generate visualizations
    _plot_roc_curve(fpr, tpr, auc, eer, eer_threshold_idx, output_dir)
    _plot_precision_recall_curve(recall_curve, precision_curve, output_dir)
    _plot_confusion_matrix(tn, fp, fn, tp, output_dir)

    return metrics


def _plot_roc_curve(fpr, tpr, auc, eer, eer_idx, output_dir):
    """Generate ROC curve."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'go', markersize=10,
             label=f'EER = {eer:.4f}')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve - Receiver Operating Characteristic', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_precision_recall_curve(recall_curve, precision_curve, output_dir):
    """Generate Precision-Recall curve."""
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_confusion_matrix(tn, fp, fn, tp, output_dir):
    """Generate confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
    cm = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'], fontsize=12)
    plt.yticks(tick_marks, ['Real', 'Fake'], fontsize=12)

    # Text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
#  RESULTS OUTPUT
# ==============================================================================

def print_results(metrics: dict) -> None:
    """Formatted output of results to console."""
    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nDataset Statistics:")
    print(f"  Total Samples:    {metrics['total_samples']}")
    print(f"  Real Images:      {metrics['negative_samples']}")
    print(f"  Fake Images:      {metrics['positive_samples']}")

    print(f"\n{'-' * 70}")
    print("  PRIMARY METRICS")
    print(f"{'-' * 70}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  AUC (ROC):        {metrics['auc']:.4f}")
    print(f"  EER:              {metrics['eer']:.4f}")

    print(f"\n{'-' * 70}")
    print("  DETECTION METRICS")
    print(f"{'-' * 70}")
    print(f"  Precision:        {metrics['precision']:.4f}  (of detected fakes, % actually fake)")
    print(f"  Recall:           {metrics['recall']:.4f}  (% of fakes detected)")
    print(f"  F1-Score:         {metrics['f1_score']:.4f}  (harmonic mean)")
    print(f"  Specificity:      {metrics['specificity']:.4f}  (% of reals correctly identified)")

    cm = metrics['confusion_matrix']
    print(f"\n{'-' * 70}")
    print("  CONFUSION MATRIX")
    print(f"{'-' * 70}")
    print(f"  True Negatives:   {cm['TN']}  (Real correctly identified)")
    print(f"  False Positives:  {cm['FP']}  (Real wrongly marked as Fake)")
    print(f"  False Negatives:  {cm['FN']}  (Fake wrongly marked as Real)")
    print(f"  True Positives:   {cm['TP']}  (Fake correctly detected)")

    print(f"\n{'-' * 70}")
    print("  INTERPRETATION")
    print(f"{'-' * 70}")

    # Recall interpretation
    total_fakes = cm['FN'] + cm['TP']
    if total_fakes > 0:
        if metrics['recall'] < 0.7:
            print(f"  [!] WARNING: Low recall ({metrics['recall']:.2%})!")
            print(f"      {cm['FN']} fakes ({cm['FN']/total_fakes*100:.1f}%) were NOT detected!")
        elif metrics['recall'] < 0.85:
            print(f"  [i] Moderate recall ({metrics['recall']:.2%})")
            print(f"      {cm['FN']} fakes still undetected")
        else:
            print(f"  [+] Good recall ({metrics['recall']:.2%})")
            print(f"      Most fakes are being detected")

    # AUC interpretation
    if metrics['auc'] < 0.6:
        print(f"  [!] Low AUC ({metrics['auc']:.4f}) - model struggles to distinguish classes")
    elif metrics['auc'] < 0.8:
        print(f"  [i] Moderate AUC ({metrics['auc']:.4f}) - acceptable performance")
    elif metrics['auc'] < 0.9:
        print(f"  [+] Good AUC ({metrics['auc']:.4f})")
    else:
        print(f"  [*] Excellent AUC ({metrics['auc']:.4f})")

    print(f"\n" + "=" * 70)


# ==============================================================================
#  MAIN FUNCTION
# ==============================================================================

def main():
    """Main function of the script."""
    args = parse_arguments()

    # Load configuration
    with open(args.detector_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    test_config_path = os.path.join(PROJECT_ROOT, 'training', 'config', 'test_config.yaml')
    with open(test_config_path, 'r', encoding='utf-8') as f:
        config2 = yaml.safe_load(f)

    # Merge configurations
    config.update(config2)

    # Override with arguments
    config['test_dataset'] = args.test_dataset
    config['weights_path'] = args.weights_path

    # Header
    print("=" * 70)
    print("  DEEPFAKE DETECTOR EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model:            {config['model_name']}")
    print(f"  Weights:          {args.weights_path}")
    print(f"  Dataset:          {args.test_dataset}")
    print(f"  Device:           {DEVICE}")
    print(f"  Resume:           {args.resume}")
    print("=" * 70)

    # Prepare data
    test_set = DeepfakeAbstractBaseDataset(config=config, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=0,  # For Windows compatibility
        collate_fn=test_set.collate_fn,
        drop_last=False
    )

    print(f"\nDataset loaded: {len(test_set)} samples")
    print(f"Batch size: {config['test_batchSize']}")
    print(f"Total batches: {len(test_loader)}")

    # Load model
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(DEVICE)

    ckpt = torch.load(args.weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    print(f"\n[OK] Model loaded successfully")

    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir, args.test_dataset)

    # Run evaluation
    print(f"\nStarting evaluation...")
    predictions, labels = evaluate_model(model, test_loader, checkpoint_mgr, args.resume)

    # Compute metrics
    output_dir = os.path.join(args.checkpoint_dir, f'{args.test_dataset}_results')
    metrics = compute_metrics(predictions, labels, output_dir)

    # Save results
    results = {
        'predictions': predictions,
        'labels': labels,
        'metrics': metrics,
        'config': config
    }
    checkpoint_mgr.save_final_results(results)

    # Print results
    print_results(metrics)

    print(f"\n[i] Results saved to: {output_dir}/")
    print(f"   - metrics.json")
    print(f"   - roc_curve.png")
    print(f"   - precision_recall_curve.png")
    print(f"   - confusion_matrix.png")
    print("\n[OK] Evaluation completed successfully!")


if __name__ == '__main__':
    main()
