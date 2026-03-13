# -*- coding: utf-8 -*-
"""
================================================================================
AIDE DETECTOR EVALUATION SCRIPT
================================================================================
Author: Diploma Thesis Project
Description: Comprehensive testing of AIDE detector on custom dataset
             with checkpoint support and metrics generation.

AIDE (AI-generated Image DEtector) is a state-of-the-art detector that
leverages hybrid features combining visual artifacts and noise patterns
for AI-generated image detection.

This script enables:
  - Testing pretrained AIDE model on image dataset
  - Computing complete metrics (AUC, Accuracy, Precision, Recall, F1)
  - Generating visualizations (ROC curve, Confusion Matrix, PR curve)
  - Saving results for further analysis

Usage:
  python evaluate_detector.py --eval_data_path <dataset_path> \
                              --checkpoint <model.pth> \
                              [--output_dir <dir>]

Reference:
  Yan et al. "A Sanity Check for AI-generated Image Detection" (ICLR 2025)
  https://github.com/shilinyan99/AIDE
================================================================================
"""

import argparse
import os
import sys
import json
import csv
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from PIL import Image

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
from data.datasets import transform_before_test, transform_train
from data.dct import DCT_base_Rec_Module


# ==============================================================================
#  CONSTANTS
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Supported folder naming conventions for real/fake images
REAL_FOLDER_NAMES = ['0_real', 'real']
FAKE_FOLDER_NAMES = ['1_fake', 'fake']


# ==============================================================================
#  HELPER CLASSES
# ==============================================================================

class DatasetArgs:
    """Container for dataset arguments required by TestDataset."""
    def __init__(self, eval_data_path):
        self.data_path = eval_data_path
        self.eval_data_path = eval_data_path


class FlexibleTestDataset(torch.utils.data.Dataset):
    """
    Test dataset that accepts both 'real/fake' and '0_real/1_fake' folder naming.
    Uses the same transforms as the original TestDataset.
    """
    def __init__(self, root):
        self.data_list = []
        self.dct = DCT_base_Rec_Module()

        # Detect folder naming convention
        real_dir = None
        fake_dir = None
        for name in REAL_FOLDER_NAMES:
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate):
                real_dir = candidate
                break
        for name in FAKE_FOLDER_NAMES:
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate):
                fake_dir = candidate
                break

        if real_dir is None or fake_dir is None:
            raise ValueError(
                f"Dataset must contain real and fake subdirectories.\n"
                f"Accepted names: {REAL_FOLDER_NAMES} and {FAKE_FOLDER_NAMES}\n"
                f"Found in {root}: {os.listdir(root)}"
            )

        for image_name in os.listdir(real_dir):
            self.data_list.append({
                'image_path': os.path.join(real_dir, image_name),
                'label': 0
            })
        for image_name in os.listdir(fake_dir):
            self.data_list.append({
                'image_path': os.path.join(fake_dir, image_name),
                'label': 1
            })

        print(f"  Real images: {sum(1 for d in self.data_list if d['label'] == 0)} (from {os.path.basename(real_dir)}/)")
        print(f"  Fake images: {sum(1 for d in self.data_list if d['label'] == 1)} (from {os.path.basename(fake_dir)}/)")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')
        image = transform_before_test(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin)
        x_maxmax = transform_train(x_maxmax)
        x_minmin1 = transform_train(x_minmin1)
        x_maxmax1 = transform_train(x_maxmax1)

        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))


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

def evaluate_model(model, dataloader, dataset, device):
    """
    Evaluate model on dataset.

    Args:
        model: AIDE model
        dataloader: DataLoader with test data
        dataset: TestDataset instance (used to retrieve file paths)
        device: Torch device

    Returns:
        Dictionary with predictions, labels, probabilities and file paths
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("Running evaluation...")
    sample_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            sample_idx += len(labels)

    # Collect file paths from dataset (order matches dataloader with shuffle=False)
    all_paths = [item['image_path'] for item in dataset.data_list[:sample_idx]]

    return {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'image_paths': all_paths
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


def save_per_image_csv(results, output_path):
    """
    Save per-image results to CSV for threshold analysis.

    Each row contains: image filename, true label, predicted probability,
    prediction at default threshold (0.5), and source folder (real/fake).
    This allows recomputing metrics at any threshold without re-running the model.

    Args:
        results: Dictionary with labels, predictions, probabilities, image_paths
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'image_name', 'true_label', 'true_class',
            'probability_fake', 'prediction_default', 'predicted_class'
        ])
        for i in range(len(results['labels'])):
            img_path = results['image_paths'][i]
            img_name = os.path.basename(img_path)
            true_label = int(results['labels'][i])
            true_class = 'fake' if true_label == 1 else 'real'
            prob_fake = float(results['probabilities'][i])
            pred = int(results['predictions'][i])
            pred_class = 'fake' if pred == 1 else 'real'
            writer.writerow([
                img_path, img_name, true_label, true_class,
                prob_fake, pred, pred_class
            ])
    print(f"Saved per-image results: {output_path}")


def save_results(metrics, results, output_dir, checkpoint_name, dataset_name):
    """Save all results to output directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(output_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save metrics as JSON
    metrics_path = os.path.join(result_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Save per-image CSV (enables threshold re-analysis)
    csv_path = os.path.join(result_dir, 'per_image_results.csv')
    save_per_image_csv(results, csv_path)

    # Save full results as pickle
    results_path = os.path.join(result_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {results_path}")

    # Generate plots
    plot_confusion_matrix(metrics, os.path.join(result_dir, 'confusion_matrix.png'))
    plot_roc_curve(results, metrics, os.path.join(result_dir, 'roc_curve.png'))
    plot_precision_recall_curve(results, metrics, os.path.join(result_dir, 'precision_recall_curve.png'))

    # Save checkpoint info
    checkpoint_path = os.path.join(result_dir, 'checkpoint_info.pkl')
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
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of images for quick testing (e.g. 50)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate dataset path
    if not os.path.exists(args.eval_data_path):
        print(f"ERROR: Dataset path does not exist: {args.eval_data_path}")
        sys.exit(1)

    # Detect folder naming convention
    has_0_real = os.path.isdir(os.path.join(args.eval_data_path, '0_real'))
    has_real = os.path.isdir(os.path.join(args.eval_data_path, 'real'))
    if not (has_0_real or has_real):
        print(f"ERROR: Dataset must contain 'real' (or '0_real') and 'fake' (or '1_fake') subdirectories")
        print(f"Found: {os.listdir(args.eval_data_path)}")
        sys.exit(1)

    # Create dataset - use FlexibleTestDataset for any naming convention
    print(f"\nLoading dataset from: {args.eval_data_path}")
    dataset = FlexibleTestDataset(args.eval_data_path)

    # Limit dataset size for quick testing
    if args.max_samples and args.max_samples < len(dataset):
        dataset.data_list = dataset.data_list[:args.max_samples]
        print(f"LIMITED to {args.max_samples} images (quick test mode)")

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
    results = evaluate_model(model, dataloader, dataset, device)

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    print_results(metrics)

    # Save results
    dataset_name = os.path.basename(os.path.normpath(args.eval_data_path))
    checkpoint_name = os.path.basename(args.checkpoint) if args.checkpoint else "none"
    result_dir = save_results(metrics, results, args.output_dir, checkpoint_name, dataset_name)
    print(f"\nAll results saved to: {result_dir}")


if __name__ == '__main__':
    main()
