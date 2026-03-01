import argparse
import os
import sys
import json
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
    confusion_matrix
)
from torch.utils.data import Dataset
from PIL import Image
import random
from tqdm import tqdm

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

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "bmp", "webp"]


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
    """
    Recursively find all image files in directory.

    Args:
        directory: Root directory to search
        extensions: List of valid file extensions

    Returns:
        List of absolute paths to image files
    """
    images = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            ext = filename.split('.')[-1].lower()
            if ext in extensions:
                images.append(os.path.join(root, filename))
    return sorted(images)


# ==============================================================================
#  DATASET CLASS
# ==============================================================================

class ImageDataset(Dataset):
    """
    Dataset for loading real and fake images for evaluation.

    Attributes:
        real_images: List of paths to real images
        fake_images: List of paths to fake images
        transform: Image preprocessing pipeline
    """

    def __init__(self, real_path, fake_path, max_samples=None, architecture="clip"):
        """
        Initialize dataset.

        Args:
            real_path: Path to directory with real images
            fake_path: Path to directory with fake images
            max_samples: Maximum number of samples per class (None = all)
            architecture: Model architecture for normalization ("clip" or "imagenet")
        """
        # Load image paths
        self.real_images = find_images(real_path)
        self.fake_images = find_images(fake_path)

        print(f"Found {len(self.real_images)} real images")
        print(f"Found {len(self.fake_images)} fake images")

        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            random.shuffle(self.real_images)
            random.shuffle(self.fake_images)
            self.real_images = self.real_images[:min(max_samples, len(self.real_images))]
            self.fake_images = self.fake_images[:min(max_samples, len(self.fake_images))]

        # Combine and create labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

        # Setup preprocessing
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
        """Load and preprocess single image."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label


# ==============================================================================
#  EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_model(model, dataloader, device):
    """
    Run model evaluation on dataset.

    Args:
        model: CLIP detector model
        dataloader: DataLoader with test images
        device: Torch device (cuda/cpu)

    Returns:
        Dictionary with labels and predictions
    """
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            # Get model predictions (sigmoid for probability)
            outputs = model(images)
            scores = outputs.sigmoid().flatten()

            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    return {
        'labels': np.array(all_labels),
        'scores': np.array(all_scores)
    }


def compute_metrics(results, threshold=0.5):
    """
    Compute comprehensive evaluation metrics.

    Args:
        results: Dictionary with 'labels' and 'scores'
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary with all computed metrics
    """
    labels = results['labels']
    scores = results['scores']
    predictions = (scores >= threshold).astype(int)

    # Basic classification metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    # Probability-based metrics
    try:
        auc_roc = roc_auc_score(labels, scores)
    except ValueError:
        auc_roc = 0.0

    try:
        avg_precision = average_precision_score(labels, scores)
    except ValueError:
        avg_precision = 0.0

    # Specificity (True Negative Rate)
    tn = ((labels == 0) & (predictions == 0)).sum()
    fp = ((labels == 0) & (predictions == 1)).sum()
    fn = ((labels == 1) & (predictions == 0)).sum()
    tp = ((labels == 1) & (predictions == 1)).sum()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'auc_roc': float(auc_roc),
        'average_precision': float(avg_precision),
        'threshold': threshold,
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'sample_counts': {
            'total': len(labels),
            'real': int((labels == 0).sum()),
            'fake': int((labels == 1).sum())
        }
    }


def print_results(metrics):
    """Print formatted evaluation results to console."""
    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS")
    print("=" * 70)

    sc = metrics['sample_counts']
    print(f"Dataset: {sc['total']} images ({sc['real']} real, {sc['fake']} fake)")
    print("-" * 70)

    print("\nClassification Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']*100:6.2f}%")
    print(f"  Precision:          {metrics['precision']*100:6.2f}%")
    print(f"  Recall:             {metrics['recall']*100:6.2f}%")
    print(f"  F1-Score:           {metrics['f1_score']*100:6.2f}%")
    print(f"  Specificity:        {metrics['specificity']*100:6.2f}%")

    print("\nProbabilistic Metrics:")
    print(f"  AUC-ROC:            {metrics['auc_roc']*100:6.2f}%")
    print(f"  Average Precision:  {metrics['average_precision']*100:6.2f}%")

    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print("                    Predicted")
    print("                  Real    Fake")
    print(f"  Actual Real    {cm['true_negative']:5d}   {cm['false_positive']:5d}")
    print(f"  Actual Fake    {cm['false_negative']:5d}   {cm['true_positive']:5d}")

    print("\n" + "=" * 70)


def save_results(metrics, output_dir, dataset_name="MyDataset"):
    """
    Save evaluation results to JSON file.

    Args:
        metrics: Dictionary with computed metrics
        output_dir: Output directory path
        dataset_name: Name for result files

    Returns:
        Path to saved results file
    """
    # Create results subdirectory
    result_subdir = os.path.join(output_dir, f"{dataset_name}_clip_results")
    os.makedirs(result_subdir, exist_ok=True)

    evaluation_date = datetime.now().strftime("%Y-%m-%d")

    # Build metrics JSON in same format as AIDE
    cm = metrics['confusion_matrix']
    sc = metrics['sample_counts']

    output_data = {
        "accuracy": round(metrics['accuracy'], 4),
        "precision": round(metrics['precision'], 4),
        "recall": round(metrics['recall'], 4),
        "f1_score": round(metrics['f1_score'], 4),
        "real_accuracy": round(metrics['specificity'], 4),
        "fake_accuracy": round(metrics['recall'], 4),
        "auc_roc": round(metrics['auc_roc'], 4),
        "average_precision": round(metrics['average_precision'], 4),
        "confusion_matrix": [
            [cm['true_negative'], cm['false_positive']],
            [cm['false_negative'], cm['true_positive']]
        ],
        "num_samples": sc['total'],
        "num_real": sc['real'],
        "num_fake": sc['fake'],
        "model": "CLIP-ViT-L/14",
        "checkpoint": "fc_weights.pth",
        "dataset": dataset_name,
        "evaluation_date": evaluation_date,
        "notes": "Evaluation of CLIP-based Universal Fake Detector on custom dataset"
    }

    # Save JSON
    output_path = os.path.join(result_subdir, "metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_subdir}")
    return output_path


# ==============================================================================
#  MAIN FUNCTION
# ==============================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='CLIP-based Universal Fake Image Detector Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_clip_detector.py \\
      --real_path C:/MyDataset/real \\
      --fake_path C:/MyDataset/fake \\
      --output_dir results
        """
    )

    parser.add_argument('--real_path', type=str, required=True,
                        help='Path to directory with real images')
    parser.add_argument('--fake_path', type=str, required=True,
                        help='Path to directory with fake images')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples per class (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--checkpoint', type=str,
                        default='../pretrained_weights/fc_weights.pth',
                        help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='vysledky',
                        help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    args = parser.parse_args()

    # Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("         CLIP-BASED UNIVERSAL FAKE IMAGE DETECTOR")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Real images: {args.real_path}")
    print(f"Fake images: {args.fake_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(args.real_path):
        print(f"ERROR: Real path not found: {args.real_path}")
        sys.exit(1)
    if not os.path.exists(args.fake_path):
        print(f"ERROR: Fake path not found: {args.fake_path}")
        sys.exit(1)

    # Load model
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
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")

    model = model.to(device)
    print(f"Model ready on {device}")

    # Create dataset
    print("\nLoading dataset...")
    dataset = ImageDataset(
        args.real_path,
        args.fake_path,
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

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(results, threshold=args.threshold)

    # Output results
    print_results(metrics)
    save_results(metrics, args.output_dir)

    print("\n[DONE] Evaluation completed successfully")


if __name__ == '__main__':
    main()
