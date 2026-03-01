import argparse
import os
import sys
import json
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
#  PATH CONFIGURATION
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
AIDE_DIR = os.path.join(PROJECT_ROOT, 'AIDE')
sys.path.insert(0, AIDE_DIR)
sys.path.insert(0, PROJECT_ROOT)

import models.AIDE as AIDE
from data.datasets import TestDataset


# ==============================================================================
#  CONSTANTS
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
#  HELPER CLASSES
# ==============================================================================

class DatasetArgs:
    """Container for dataset arguments required by TestDataset."""
    def __init__(self, eval_data_path):
        self.data_path = eval_data_path
        self.eval_data_path = eval_data_path


# ==============================================================================
#  MODEL FUNCTIONS
# ==============================================================================

def create_model(resnet_path, convnext_path, device):
    """
    Create and initialize AIDE model.

    Args:
        resnet_path: Path to pretrained ResNet checkpoint
        convnext_path: Path to pretrained ConvNeXt checkpoint
        device: Torch device (cuda or cpu)

    Returns:
        Initialized AIDE model
    """
    print("Creating AIDE model...")
    model = AIDE.AIDE(resnet_path=resnet_path, convnext_path=convnext_path)
    model = model.to(device)
    print("Model created successfully")
    return model


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint.

    Args:
        model: AIDE model instance
        checkpoint_path: Path to checkpoint file
        device: Torch device

    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return model

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print("Checkpoint loaded successfully")
    return model


# ==============================================================================
#  EVALUATION
# ==============================================================================

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.

    Args:
        model: AIDE model
        dataloader: DataLoader with test data
        device: Torch device

    Returns:
        Dictionary with predictions and labels
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities)
    }


# ==============================================================================
#  METRICS COMPUTATION
# ==============================================================================

def compute_metrics(results):
    """
    Compute evaluation metrics.

    Args:
        results: Dictionary with labels, predictions, probabilities

    Returns:
        Dictionary with computed metrics
    """
    labels = results['labels']
    predictions = results['predictions']
    probabilities = results['probabilities']

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Per-class accuracy
    real_mask = labels == 0
    fake_mask = labels == 1
    real_accuracy = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0
    fake_accuracy = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0

    # ROC and PR metrics
    try:
        auc_roc = roc_auc_score(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)
    except ValueError:
        auc_roc = 0.0
        avg_precision = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'real_accuracy': float(real_accuracy),
        'fake_accuracy': float(fake_accuracy),
        'auc_roc': float(auc_roc),
        'average_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(labels),
        'num_real': int(real_mask.sum()),
        'num_fake': int(fake_mask.sum())
    }

    return metrics


# ==============================================================================
#  VISUALIZATION
# ==============================================================================

def plot_confusion_matrix(metrics, output_path):
    """Generate and save confusion matrix plot."""
    cm = np.array(metrics['confusion_matrix'])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Real', 'Fake']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='AIDE Detector - Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curve(results, metrics, output_path):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {metrics["auc_roc"]:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('AIDE Detector - ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_precision_recall_curve(results, metrics, output_path):
    """Generate and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2,
            label=f'PR curve (AP = {metrics["average_precision"]:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('AIDE Detector - Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ==============================================================================
#  RESULTS OUTPUT
# ==============================================================================

def print_results(metrics):
    """Print evaluation results to console."""
    print("\n" + "=" * 70)
    print("                     EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total samples: {metrics['num_samples']}")
    print(f"  Real images: {metrics['num_real']}")
    print(f"  Fake images: {metrics['num_fake']}")
    print("-" * 70)
    print(f"Overall Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Real Accuracy:        {metrics['real_accuracy']:.4f} ({metrics['real_accuracy']*100:.2f}%)")
    print(f"Fake Accuracy:        {metrics['fake_accuracy']:.4f} ({metrics['fake_accuracy']*100:.2f}%)")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall:               {metrics['recall']:.4f}")
    print(f"F1 Score:             {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:              {metrics['auc_roc']:.4f}")
    print(f"Average Precision:    {metrics['average_precision']:.4f}")
    print("-" * 70)
    print("Confusion Matrix:")
    print("              Predicted")
    print("              Real  Fake")
    cm = metrics['confusion_matrix']
    print(f"Actual Real   {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       Fake   {cm[1][0]:5d} {cm[1][1]:5d}")
    print("=" * 70)


def save_results(metrics, results, output_dir, checkpoint_name):
    """Save all results to output directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(output_dir, f"MyDataset_full_results")
    os.makedirs(result_dir, exist_ok=True)

    # Save metrics as JSON
    metrics_path = os.path.join(result_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Save full results as pickle
    results_path = os.path.join(result_dir, f"MyDataset_full_{timestamp}_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {results_path}")

    # Generate plots
    plot_confusion_matrix(metrics, os.path.join(result_dir, 'confusion_matrix.png'))
    plot_roc_curve(results, metrics, os.path.join(result_dir, 'roc_curve.png'))
    plot_precision_recall_curve(results, metrics, os.path.join(result_dir, 'precision_recall_curve.png'))

    # Save checkpoint info
    checkpoint_path = os.path.join(result_dir, f"MyDataset_full_{timestamp}_checkpoint.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'checkpoint': checkpoint_name, 'timestamp': timestamp}, f)

    return result_dir


# ==============================================================================
#  MAIN FUNCTION
# ==============================================================================

def main():
    """Main function for AIDE detector evaluation."""
    parser = argparse.ArgumentParser(
        description='AIDE Detector Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_detector.py --eval_data_path C:/MyDataset_prepared \
                              --checkpoint ../AIDE/checkpoints/GenImage_train.pth

  # With all backbone weights
  python evaluate_detector.py --eval_data_path C:/MyDataset_prepared \
                              --checkpoint ../AIDE/checkpoints/GenImage_train.pth \
                              --resnet_path ../AIDE/checkpoints/resnet50.pth \
                              --convnext_path ../AIDE/checkpoints/open_clip_pytorch_model.bin
        """
    )
    parser.add_argument('--eval_data_path', required=True, type=str,
                        help='Path to evaluation dataset (with 0_real and 1_fake folders)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to AIDE model checkpoint')
    parser.add_argument('--resnet_path', type=str, default=None,
                        help='Path to pretrained ResNet checkpoint')
    parser.add_argument('--convnext_path', type=str, default=None,
                        help='Path to pretrained ConvNeXt checkpoint')
    parser.add_argument('--output_dir', type=str, default='vysledky',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate dataset path
    if not os.path.exists(args.eval_data_path):
        print(f"ERROR: Dataset path does not exist: {args.eval_data_path}")
        sys.exit(1)

    real_path = os.path.join(args.eval_data_path, '0_real')
    fake_path = os.path.join(args.eval_data_path, '1_fake')
    if not (os.path.exists(real_path) and os.path.exists(fake_path)):
        print(f"ERROR: Dataset must contain '0_real' and '1_fake' subdirectories")
        print(f"Run prepare_dataset.py first to prepare your dataset")
        sys.exit(1)

    # Create dataset
    print(f"\nLoading dataset from: {args.eval_data_path}")
    dataset_args = DatasetArgs(args.eval_data_path)
    dataset = TestDataset(is_train=False, args=dataset_args)
    print(f"Dataset loaded: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Create and load model
    model = create_model(args.resnet_path, args.convnext_path, device)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, device)

    # Run evaluation
    results = evaluate_model(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print_results(metrics)

    # Save results
    checkpoint_name = os.path.basename(args.checkpoint) if args.checkpoint else "none"
    result_dir = save_results(metrics, results, args.output_dir, checkpoint_name)
    print(f"\nAll results saved to: {result_dir}")


if __name__ == '__main__':
    main()
