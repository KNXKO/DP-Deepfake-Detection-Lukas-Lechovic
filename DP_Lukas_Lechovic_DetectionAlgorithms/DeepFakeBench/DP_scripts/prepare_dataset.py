import json
import os
import glob
import sys
import argparse
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Dataset JSON Generator for DeepfakeBench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_dataset.py --dataset_path C:/Datasets/MyDeepfakes

  # With custom name and output path
  python prepare_dataset.py --dataset_path C:/Datasets/MyDeepfakes \\
                            --dataset_name CustomDataset \\
                            --output_path ./custom_dataset.json
        """
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='C:/Users/diabo/Desktop/MyDataset',
        help='Path to dataset root directory (default: C:/Users/diabo/Desktop/MyDataset)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path for output JSON file (default: preprocessing/dataset_json/<name>.json)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='MyDataset_full',
        help='Dataset name in JSON structure (default: MyDataset_full)'
    )
    return parser.parse_args()


def find_images(directory: str) -> list:
    """
    Find all images in directory.

    Args:
        directory: Path to directory

    Returns:
        list: List of image paths
    """
    images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']

    for ext in extensions:
        images.extend(glob.glob(os.path.join(directory, ext)))
        # Recursive search in subdirectories
        images.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    return sorted(list(set(images)))  # Remove duplicates and sort


def create_dataset_json(dataset_path: str, dataset_name: str, output_path: str) -> dict:
    """
    Create JSON configuration file for dataset.

    Args:
        dataset_path: Path to dataset root directory
        dataset_name: Dataset name
        output_path: Path for output JSON

    Returns:
        dict: Dataset statistics
    """
    real_path = os.path.join(dataset_path, "real")
    fake_path = os.path.join(dataset_path, "fake")

    # Validation
    if not os.path.exists(real_path):
        print(f"[ERROR] Folder 'real' does not exist: {real_path}")
        sys.exit(1)
    if not os.path.exists(fake_path):
        print(f"[ERROR] Folder 'fake' does not exist: {fake_path}")
        sys.exit(1)

    # Find images
    real_images = find_images(real_path)
    fake_images = find_images(fake_path)

    print("=" * 70)
    print("    GENERATING DATASET JSON FILE")
    print("=" * 70)
    print(f"\nDataset source: {dataset_path}")
    print(f"\nFound images:")
    print(f"  Real: {len(real_images)} images")
    print(f"  Fake: {len(fake_images)} images")
    print(f"  TOTAL: {len(real_images) + len(fake_images)} images")
    print()

    if len(real_images) == 0:
        print("[ERROR] No images found in 'real' folder!")
        sys.exit(1)
    if len(fake_images) == 0:
        print("[ERROR] No images found in 'fake' folder!")
        sys.exit(1)

    # Create JSON structure compatible with DeepfakeBench
    json_structure = {
        dataset_name: {
            f"{dataset_name.replace('_full', '')}_real": {
                "test": {}
            },
            f"{dataset_name.replace('_full', '')}_fake": {
                "test": {}
            }
        }
    }

    video_counter = 0

    # Add real images
    print(f"Processing {len(real_images)} REAL images...")
    real_label = f"{dataset_name.replace('_full', '')}_real"
    for img_path in real_images:
        video_name = f"video_{video_counter:05d}"
        video_counter += 1
        abs_path = os.path.abspath(img_path).replace('\\', '/')

        json_structure[dataset_name][real_label]["test"][video_name] = {
            "label": real_label,
            "frames": [abs_path]
        }

    # Add fake images
    print(f"Processing {len(fake_images)} FAKE images...")
    fake_label = f"{dataset_name.replace('_full', '')}_fake"
    for img_path in fake_images:
        video_name = f"video_{video_counter:05d}"
        video_counter += 1
        abs_path = os.path.abspath(img_path).replace('\\', '/')

        json_structure[dataset_name][fake_label]["test"][video_name] = {
            "label": fake_label,
            "frames": [abs_path]
        }

    # Backup existing file
    if os.path.exists(output_path):
        backup_path = output_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"[i] Existing file backed up: {backup_path}")

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, indent=2, ensure_ascii=False)

    # Statistics
    stats = {
        'total': len(real_images) + len(fake_images),
        'real': len(real_images),
        'fake': len(fake_images),
        'real_percentage': len(real_images) / (len(real_images) + len(fake_images)) * 100,
        'fake_percentage': len(fake_images) / (len(real_images) + len(fake_images)) * 100
    }

    print()
    print(f"[OK] JSON file created: {output_path}")
    print()
    print("=" * 70)
    print("    DATASET STATISTICS")
    print("=" * 70)
    print(f"  Real:  {stats['real']:>6} images ({stats['real_percentage']:.1f}%)")
    print(f"  Fake:  {stats['fake']:>6} images ({stats['fake_percentage']:.1f}%)")
    print(f"  TOTAL: {stats['total']:>6} images")
    print("=" * 70)

    return stats


def main():
    """Main function."""
    args = parse_arguments()

    # Determine output path
    if args.output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(
            project_root,
            'preprocessing',
            'dataset_json',
            f'{args.dataset_name}.json'
        )
    else:
        output_path = args.output_path

    # Generate JSON
    stats = create_dataset_json(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output_path=output_path
    )

    print(f"\n[OK] Dataset ready for testing!")
    print(f"\nTo run evaluation use:")
    print(f"  python evaluate_detector.py --detector_path <config.yaml> \\")
    print(f"                              --weights_path <model.pth> \\")
    print(f"                              --test_dataset {args.dataset_name}")


if __name__ == '__main__':
    main()
