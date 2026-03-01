import os
import json
import glob
from pathlib import Path

def create_dataset_json_direct():
    """Vytvorí JSON súbor priamo z originálnych obrázkov"""

    # Cesty k dátam - používame originálne obrázky
    dataset_root = Path("../datasets/rgb/Celeb-DF-v1")
    real_path = dataset_root / "Celeb-real"
    fake_path = dataset_root / "Celeb-synthesis"

    # Inicializácia štruktúry JSON
    dataset_json = {
        "Celeb-DF-v1": {
            "CelebDFv1_real": {
                "train": {},
                "val": {},
                "test": {}
            },
            "CelebDFv1_fake": {
                "train": {},
                "val": {},
                "test": {}
            }
        }
    }

    def get_all_images(path):
        """Získa všetky obrázky v priečinku a podpriečinkoch"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(glob.glob(str(path / '**' / ext), recursive=True))
        return [str(Path(img).resolve()) for img in images]

    def process_images_for_label(image_list, label, dataset_json):
        """Spracuje zoznam obrázkov a rozdelí ich na train/val/test"""
        total = len(image_list)
        train_split = int(0.7 * total)
        val_split = int(0.85 * total)

        # Vytvor "video" názvy z obrázkov - zoskup po 10 obrázkov do jedného "videa"
        video_count = 0
        for i in range(0, total, 10):  # Každých 10 obrázkov = 1 "video"
            video_name = f"video_{video_count:05d}"
            frames = image_list[i:i+10]  # Max 10 obrázkov na "video"

            # Rozhodni o train/val/test split
            if i < train_split:
                split = "train"
            elif i < val_split:
                split = "val"
            else:
                split = "test"

            dataset_json["Celeb-DF-v1"][label][split][video_name] = {
                "label": label,
                "frames": frames
            }

            video_count += 1

    # Spracovanie reálnych obrázkov
    if real_path.exists():
        real_images = get_all_images(real_path)
        print(f"Našiel som {len(real_images)} reálnych obrázkov")
        process_images_for_label(real_images, "CelebDFv1_real", dataset_json)

    # Spracovanie falošných obrázkov
    if fake_path.exists():
        fake_images = get_all_images(fake_path)
        print(f"Našiel som {len(fake_images)} falošných obrázkov")
        process_images_for_label(fake_images, "CelebDFv1_fake", dataset_json)

    # Vytvor priečinok pre JSON súbory
    json_dir = Path("dataset_json")
    json_dir.mkdir(exist_ok=True)

    # Ulož JSON súbor
    output_path = json_dir / "Celeb-DF-v1.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"JSON súbor uložený do: {output_path}")

    # Výpis štatistík
    for label in ["CelebDFv1_real", "CelebDFv1_fake"]:
        train_count = len(dataset_json["Celeb-DF-v1"][label]["train"])
        val_count = len(dataset_json["Celeb-DF-v1"][label]["val"])
        test_count = len(dataset_json["Celeb-DF-v1"][label]["test"])

        print(f"{label}:")
        print(f"  Train videos: {train_count}")
        print(f"  Val videos: {val_count}")
        print(f"  Test videos: {test_count}")

if __name__ == "__main__":
    create_dataset_json_direct()