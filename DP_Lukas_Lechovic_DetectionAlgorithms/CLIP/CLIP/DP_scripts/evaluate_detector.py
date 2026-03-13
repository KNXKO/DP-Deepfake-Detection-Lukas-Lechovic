# -*- coding: utf-8 -*-
"""
================================================================================
CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR - EVALUATION SCRIPT
================================================================================
Author: Diploma Thesis Project
Description: Comprehensive evaluation of CLIP-based detector on custom dataset
             with per-image CSV logging, full metrics, visualizations,
             and threshold analysis support.

Usage:
  python evaluate_detector.py --real_path <real_images> --fake_path <fake_images>
  python evaluate_detector.py --dataset_path <dataset_root>

Reference:
  Ojha et al. "Towards Universal Fake Image Detectors that Generalize Across
  Generative Models" (CVPR 2023)
================================================================================
"""

import argparse
import os
import sys
import csv
import json
import pickle
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from torch.utils.data import Dataset
from PIL import Image
import random
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models import get_model


# ==============================================================================
#  CONSTANTS
# ==============================================================================

SEED = 42

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


# ==============================================================================
#  UTILITY FUNCTIONS
# ==============================================================================

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_images(directory, extensions=SUPPORTED_EXTENSIONS):
    """Recursively find all image files in directory."""
    images = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext in extensions:
                images.append(os.path.join(root, filename))
    return sorted(images)


def detect_dataset_folders(dataset_path):
    """
    Auto-detect real/fake folder naming convention.

    Supports:
      - real/ fake/       (alternative)
      - 0_real/ 1_fake/   (standard)

    Returns:
        (real_path, fake_path) or raises error
    """
    candidates = [
        ("real", "fake"),
        ("0_real", "1_fake"),
    ]

    for real_name, fake_name in candidates:
        real_path = os.path.join(dataset_path, real_name)
        fake_path = os.path.join(dataset_path, fake_name)
        if os.path.isdir(real_path) and os.path.isdir(fake_path):
            print(f"Auto-detected folders: {real_name}/ + {fake_name}/")
            return real_path, fake_path

    available = [d for d in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, d))]
    raise FileNotFoundError(
        f"Could not find real/fake folders in {dataset_path}.\n"
        f"Expected: real/+fake/ or 0_real/+1_fake/\n"
        f"Found subdirectories: {available}"
    )


# ==============================================================================
#  DATASET CLASS
# ==============================================================================

class ImageDataset(Dataset):
    """Dataset for loading real and fake images with per-image path tracking."""

    def __init__(self, real_path, fake_path, max_samples=None, architecture="clip"):
        self.real_images = find_images(real_path)
        self.fake_images = find_images(fake_path)

        print(f"Found {len(self.real_images)} real images")
        print(f"Found {len(self.fake_images)} fake images")

        if max_samples is not None and max_samples > 0:
            random.shuffle(self.real_images)
            random.shuffle(self.fake_images)
            self.real_images = self.real_images[:min(max_samples, len(self.real_images))]
            self.fake_images = self.fake_images[:min(max_samples, len(self.fake_images))]
            print(f"Limited to {len(self.real_images)} real + {len(self.fake_images)} fake (max_samples={max_samples})")

        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

        stat_key = "imagenet" if architecture.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_key], std=STD[stat_key]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label, idx
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label, idx


# ==============================================================================
#  EVALUATION
# ==============================================================================

def evaluate_model(model, dataloader, dataset, device):
    """
    Run model evaluation and collect per-image results.

    Returns:
        Dictionary with labels, predictions, probabilities, image_paths
    """
    model.eval()

    all_labels = []
    all_probabilities = []
    all_image_paths = []

    with torch.no_grad():
        for images, labels, indices in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            scores = outputs.sigmoid().flatten().cpu().numpy()
            labels_np = labels.numpy()
            indices_np = indices.numpy()

            for i in range(len(labels_np)):
                idx = int(indices_np[i])
                all_labels.append(int(labels_np[i]))
                all_probabilities.append(float(scores[i]))
                all_image_paths.append(dataset.image_paths[idx])

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probabilities)
    preds_arr = (probs_arr >= 0.5).astype(int)

    return {
        'labels': labels_arr,
        'predictions': preds_arr,
        'probabilities': probs_arr,
        'image_paths': all_image_paths
    }


# ==============================================================================
#  METRICS COMPUTATION
# ==============================================================================

def compute_metrics(results, threshold=0.5):
    """Compute comprehensive evaluation metrics."""
    labels = results['labels']
    probabilities = results['probabilities']
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Per-class accuracy
    real_mask = labels == 0
    fake_mask = labels == 1
    real_accuracy = accuracy_score(labels[real_mask], predictions[real_mask]) if real_mask.sum() > 0 else 0
    fake_accuracy = accuracy_score(labels[fake_mask], predictions[fake_mask]) if fake_mask.sum() > 0 else 0

    try:
        auc_roc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc_roc = 0.0

    try:
        avg_precision = average_precision_score(labels, probabilities)
    except ValueError:
        avg_precision = 0.0

    cm = confusion_matrix(labels, predictions)

    return {
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
           title='CLIP Detector - Confusion Matrix',
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
    ax.set_title('CLIP Detector - ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_precision_recall_curve(results, metrics, output_path):
    """Generate and save Precision-Recall curve plot."""
    precision_vals, recall_vals, _ = precision_recall_curve(
        results['labels'], results['probabilities'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color='green', lw=2,
            label=f'PR curve (AP = {metrics["average_precision"]:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('CLIP Detector - Precision-Recall Curve')
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
    """Print evaluation results to console in standardized format."""
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
    """Save per-image results to CSV for threshold analysis."""
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
    print(f"Saved: {output_path}")


def save_results(metrics, results, output_dir, checkpoint_name, dataset_name):
    """Save all results to output directory."""
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved: {metrics_path}")

    # Save per-image CSV (enables threshold re-analysis)
    csv_path = os.path.join(output_dir, 'per_image_results.csv')
    save_per_image_csv(results, csv_path)

    # Save full results as pickle
    results_path = os.path.join(output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {results_path}")

    # Generate plots
    plot_confusion_matrix(metrics, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_roc_curve(results, metrics, os.path.join(output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(results, metrics, os.path.join(output_dir, 'precision_recall_curve.png'))

    # Save checkpoint info
    checkpoint_info_path = os.path.join(output_dir, 'checkpoint_info.pkl')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(checkpoint_info_path, 'wb') as f:
        pickle.dump({'checkpoint': checkpoint_name, 'timestamp': timestamp}, f)

    return output_dir


# ==============================================================================
#  MAIN FUNCTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CLIP-based Universal Fake Image Detector Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Root path to dataset (auto-detects real/fake or 0_real/1_fake)')
    parser.add_argument('--real_path', type=str, default=None,
                        help='Path to directory with real images')
    parser.add_argument('--fake_path', type=str, default=None,
                        help='Path to directory with fake images')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per class for quick testing (e.g. 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--checkpoint', type=str,
                        default='../pretrained_weights/fc_weights.pth',
                        help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='vysledky',
                        help='Base output directory for results')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name for the dataset (auto-detected from path if not set)')

    args = parser.parse_args()

    # --- Resolve dataset paths ---
    if args.dataset_path:
        real_path, fake_path = detect_dataset_folders(args.dataset_path)
        if args.dataset_name is None:
            args.dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    elif args.real_path and args.fake_path:
        real_path = args.real_path
        fake_path = args.fake_path
        if args.dataset_name is None:
            args.dataset_name = os.path.basename(os.path.dirname(os.path.normpath(real_path)))
    else:
        parser.error("Provide either --dataset_path OR both --real_path and --fake_path")

    # --- Setup ---
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("         CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR")
    print("=" * 70)
    print(f"Device:        {device}")
    print(f"Real images:   {real_path}")
    print(f"Fake images:   {fake_path}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Dataset name:  {args.dataset_name}")
    if args.max_samples:
        print(f"Max samples:   {args.max_samples} per class (QUICK TEST MODE)")
    print("=" * 70)

    # --- Validate paths ---
    if not os.path.exists(real_path):
        print(f"ERROR: Real path not found: {real_path}")
        sys.exit(1)
    if not os.path.exists(fake_path):
        print(f"ERROR: Fake path not found: {fake_path}")
        sys.exit(1)

    # --- Create output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir:    {output_dir}")

    # --- Load model ---
    print("\nLoading CLIP model...")
    model = get_model("CLIP:ViT-L/14")

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_path)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.fc.load_state_dict(state_dict)
        print("Checkpoint loaded successfully")
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = model.to(device)
    print(f"Model ready on {device}")

    # --- Create dataset ---
    print("\nLoading dataset...")
    dataset = ImageDataset(
        real_path,
        fake_path,
        max_samples=args.max_samples,
        architecture="clip"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Total images: {len(dataset)}")

    # --- Evaluate ---
    print("\nRunning evaluation...")
    results = evaluate_model(model, dataloader, dataset, device)

    # --- Compute and display metrics ---
    metrics = compute_metrics(results)
    print_results(metrics)

    # --- Save all results ---
    checkpoint_name = os.path.basename(args.checkpoint)
    save_results(metrics, results, output_dir, checkpoint_name, args.dataset_name)

    # --- Summary ---
    print(f"\nAll results saved to: {output_dir}")
    print("Files:")
    print(f"  - metrics.json               (evaluation metrics)")
    print(f"  - per_image_results.csv      (per-image probabilities)")
    print(f"  - results.pkl                (numpy arrays for analysis)")
    print(f"  - checkpoint_info.pkl        (checkpoint metadata)")
    print(f"  - confusion_matrix.png       (confusion matrix plot)")
    print(f"  - roc_curve.png              (ROC curve)")
    print(f"  - precision_recall_curve.png (PR curve)")
    print(f"\nTo analyze different thresholds without re-running the model:")
    print(f"  python analyze_results.py --results_dir \"{output_dir}\"")
    print("\n[DONE] Evaluation completed successfully")


if __name__ == '__main__':
    main()
