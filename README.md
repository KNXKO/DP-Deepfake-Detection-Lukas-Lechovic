# Diplomová práca / Diploma Thesis

**Autor / Author:** Lukáš Lechovič
**Téma / Topic:** Generovanie vizuálnych informácií a rozpoznávanie generovaného či zmanipulovaného obsahu / Generation of Visual Information and Detection of Generated or Manipulated Content
**Vedúci práce / Supervisor:** PaedDr. Miroslav Ölvecký, PhD.

---

## SK

Repozitár obsahuje praktickú časť diplomovej práce. Zahŕňa automatizačný skript na generovanie testovacieho datasetu pomocou nástroja FaceFusion a implementáciu štyroch detekčných algoritmov testovaných na vlastnom datasete FFHQ-FaceFusion-10k.

## EN

This repository contains the practical part of the diploma thesis. It includes an automation script for generating a test dataset using the FaceFusion tool and the implementation of four detection algorithms tested on a custom dataset FFHQ-FaceFusion-10k.

---

## Štruktúra projektu / Project Structure

```
DP-PraktickaCast_Lukas-Lechovic/
│
├── DP_Lukas_Lechovic_FaceFusion/
│   └── Script_AutomatedGenerateImages/
│       └── process_male_faces.bat        # Dávkový skript na generovanie zmanipulovaných obrazov
│                                         # Batch script for generating manipulated images
│
└── DP_Lukas_Lechovic_DetectionAlgorithms/
    ├── AIDE/                             # Algoritmus AIDE / AIDE Algorithm
    │   └── AIDE/DP_scripts/             # Vlastné testovacie skripty / Custom test scripts
    ├── CLIP/                             # Algoritmus CLIP-ViT (UniversalFakeDetect)
    │   └── CLIP/DP_scripts/
    ├── CNN/                              # Algoritmus CNNDetection / CNNDetection Algorithm
    │   └── CNNDetection/DP_scripts/
    └── DeepFakeBench/                    # Algoritmus Xception (cez DeepFakeBench)
        └── DP_scripts/                  # Xception Algorithm (via DeepFakeBench)
```

Dataset nie je súčasťou repozitára kvôli veľkosti (16,1 GB). Postup stiahnutia je popísaný nižšie.
The dataset is not included in the repository due to its size (16.1 GB). Download instructions are provided below.

---

## Testovací dataset / Test Dataset – FFHQ-FaceFusion-10k

Dataset obsahuje 10 000 obrazov rozdelených rovnomerne na 5 000 autentických a 5 000 zmanipulovaných obrazov. Všetky obrazy majú rozlíšenie 1024×1024 px a sú uložené vo formáte PNG.

The dataset contains 10,000 images evenly split into 5,000 authentic and 5,000 manipulated images. All images have a resolution of 1024×1024 px and are stored in PNG format.

| Kategória / Category | Počet / Count | Veľkosť / Size | Pomenovanie / Naming |
|---|---|---|---|
| Autentické / Authentic | 5 000 | 6,26 GB | `00005.png` – `60688.png` |
| Zmanipulované / Manipulated | 5 000 | 9,92 GB | `fake_00005.png` – `fake_60688.png` |

**Zdrojové autentické obrazy / Source authentic images:** dataset [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) od NVIDIA (70 000 snímok tvárí, 1024×1024 px) / from NVIDIA (70,000 face images, 1024×1024 px).

**Štruktúra datasetu na disku / Dataset structure on disk:**
```
FFHQ-FaceFusion-10k/
├── real/     # Autentické FFHQ tváre / Authentic FFHQ faces
└── fake/     # Zmanipulované obrazy / Manipulated images (prefixed fake_)
```

### Stiahnutie datasetu / Dataset Download

Dataset FFHQ-FaceFusion-10k je dostupný na Kaggle: / The FFHQ-FaceFusion-10k dataset is available on Kaggle:

**[kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k](https://www.kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k)**

```bash
# Stiahnutie cez Kaggle CLI (vyžaduje kaggle API token)
# Download via Kaggle CLI (requires kaggle API token)
kaggle datasets download -d lukaslechovic/ffhq-facefusion-10k
```

---

## Generovanie zmanipulovaných obrazov / Generating Manipulated Images

Zmanipulované obrazy boli vygenerované automatizovaným dávkovým skriptom `process_male_faces.bat` pomocou nástroja FaceFusion v headless režime (bez grafického rozhrania).

Manipulated images were generated using an automated batch script `process_male_faces.bat` with the FaceFusion tool in headless mode (without GUI).

**Požiadavky / Requirements:**
- Nainštalovaná aplikácia [Pinokio](https://pinokio.computer/) s FaceFusion / Installed [Pinokio](https://pinokio.computer/) application with FaceFusion

**Parametre použité pri generovaní / Parameters used for generation:**

| Komponent / Component | Model | Funkcia / Function |
|---|---|---|
| Výmena tváre / Face Swap (Deep Swapper) | `iperov/elon_musk_224` | Výmena tváre pomocou hlbokého učenia / Deep learning face swap |
| Vylepšovač tváre / Face Enhancer | `gpen_bfr_1024` | Vylepšenie kvality a odstránenie artefaktov / Quality enhancement and artifact removal |
| Obnova výrazov / Expression Restorer | `live_portrait` | Zachovanie pôvodných výrazov tváre / Preserving original facial expressions |
| Výstupné rozlíšenie / Output Resolution | 1024×1024 px | – |
| Kvalita JPEG / JPEG Quality | 90 % | – |

```bash
cd DP_Lukas_Lechovic_FaceFusion/Script_AutomatedGenerateImages
process_male_faces.bat
```

Generovanie jedného obrazu trvalo v priemere ~13 sekúnd (GPU GTX 1070 8 GB). Celkový čas generovania 5 000 obrazov bol ~18 hodín.

Generating one image took ~13 seconds on average (GPU GTX 1070 8 GB). Total generation time for 5,000 images was ~18 hours.

---

## Detekčné algoritmy / Detection Algorithms

Každý algoritmus má v priečinku `DP_scripts/` vlastný skript `evaluate_detector.py` a dávkový spúšťací skript `run_evaluation.bat`. Conda prostredia sú definované v `DP_scripts/conda_env/*.yml`.

Each algorithm has its own `evaluate_detector.py` script and a batch launcher `run_evaluation.bat` in the `DP_scripts/` folder. Conda environments are defined in `DP_scripts/conda_env/*.yml`.

Spoločné parametre testovania / Common testing parameters:
- Prahová hodnota klasifikácie / Classification threshold: **0,5**
- Dataset: **10 000 obrazov / images** (5 000 real + 5 000 fake)
- Výstup / Output: JSON súbor s metrikami / JSON file with metrics (`DP_scripts/vysledky/metrics.json`)

---

### Xception (cez / via DeepFakeBench)

**Repozitár / Repository:** [github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
**Conda prostredie / Conda environment:** Python 3.8, PyTorch ≥1.13.0, CUDA

```bash
conda activate deepfakebench

# Krok 1: Príprava JSON konfigurácie datasetu
# Step 1: Prepare dataset JSON configuration
cd DP_Lukas_Lechovic_DetectionAlgorithms/DeepFakeBench/DP_scripts
python prepare_dataset.py --dataset_path C:/FFHQ-FaceFusion-10k --dataset_name MyDataset

# Krok 2: Spustenie hodnotenia / Step 2: Run evaluation
run_evaluation.bat
# alebo manuálne / or manually:
python evaluate_detector.py \
    --detector_path ../training/config/detector/xception.yaml \
    --weights_path ../training/weights/xception_best.pth \
    --test_dataset MyDataset
```

Vstupné rozlíšenie pre model / Model input resolution: 299×299 px. Čas spracovania / Processing time: ~12 hodín / hours.

---

### CLIP-ViT (UniversalFakeDetect)

**Repozitár / Repository:** [github.com/WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
**Conda prostredie / Conda environment:** Python 3.10, PyTorch ≥1.13.0, CUDA

```bash
conda activate clip

cd DP_Lukas_Lechovic_DetectionAlgorithms/CLIP/CLIP/DP_scripts
run_evaluation.bat
# alebo manuálne / or manually:
python evaluate_detector.py \
    --real_path C:/FFHQ-FaceFusion-10k/real \
    --fake_path C:/FFHQ-FaceFusion-10k/fake \
    --checkpoint ../pretrained_weights/fc_weights.pth \
    --output_dir vysledky
```

Vstupné rozlíšenie pre model / Model input resolution: 224×224 px (resize na 256, center crop). Čas spracovania / Processing time: ~11 hodín / hours.

---

### AIDE

**Repozitár / Repository:** [github.com/shilinyan99/AIDE](https://github.com/shilinyan99/AIDE)
**Conda prostredie / Conda environment:** Python 3.10, PyTorch 2.0.1, CUDA 11.8

Algoritmus vyžaduje tri checkpointy / The algorithm requires three checkpoints:
- `checkpoints/GenImage_train.pth` – hlavný AIDE checkpoint / main AIDE checkpoint
- `checkpoints/resnet50.pth` – ResNet-50
- `checkpoints/open_clip_pytorch_model.bin` – ConvNeXt (OpenCLIP)

```bash
conda activate aide

cd DP_Lukas_Lechovic_DetectionAlgorithms/AIDE/AIDE/DP_scripts
run_evaluation.bat
# alebo manuálne / or manually:
python evaluate_detector.py \
    --eval_data_path C:/FFHQ-FaceFusion-10k \
    --checkpoint ../AIDE/checkpoints/GenImage_train.pth \
    --resnet_path ../AIDE/checkpoints/resnet50.pth \
    --convnext_path ../AIDE/checkpoints/open_clip_pytorch_model.bin
```

Štruktúra datasetu pre AIDE: priečinky `0_real/` a `1_fake/` (skript konvertuje automaticky). Čas spracovania ~11 hodín.

Dataset structure for AIDE: folders `0_real/` and `1_fake/` (the script converts automatically). Processing time ~11 hours.

---

### CNNDetection

**Repozitár / Repository:** [github.com/PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection)
**Conda prostredie / Conda environment:** Python 3.10, PyTorch ≥1.2.0, CUDA

Predtrénované váhy (`blur_jpg_prob0.5.pth`) nie sú súčasťou repozitára a je ich potrebné stiahnuť manuálne podľa pokynov v `readme.md` projektu CNNDetection.

Pre-trained weights (`blur_jpg_prob0.5.pth`) are not included in the repository and must be downloaded manually according to the CNNDetection project `readme.md`.

```bash
conda activate cnn

cd DP_Lukas_Lechovic_DetectionAlgorithms/CNN/CNNDetection/DP_scripts
run_evaluation.bat
# alebo manuálne / or manually:
python evaluate_detector.py \
    -d C:/FFHQ-FaceFusion-10k \
    -m ../weights/blur_jpg_prob0.5.pth \
    --threshold 0.5
```

Model: ResNet-50 trénovaný na ProGAN obrazoch / trained on ProGAN images. Vstupné rozlíšenie pre model / Model input resolution: 224×224 px. Čas spracovania / Processing time: ~14 hodín / hours.

---

## Výstup hodnotenia / Evaluation Output

Každý algoritmus ukladá výsledky do `DP_scripts/vysledky/`:
Each algorithm saves results to `DP_scripts/vysledky/`:

| Súbor / File | Obsah / Content |
|---|---|
| `metrics.json` | Celková presnosť, presnosť, citlivosť, F1-skóre, špecifickosť, AUC-ROC, matica zámen / Overall accuracy, precision, recall, F1-score, specificity, AUC-ROC, confusion matrix |
| `confusion_matrix.png` | Vizualizácia matice zámen / Confusion matrix visualization |
| `roc_curve.png` | ROC krivka / ROC curve |
| `precision_recall_curve.png` | Precision-Recall krivka / Precision-Recall curve |

---

## Požiadavky / Requirements

- Anaconda / Miniconda
- NVIDIA GPU s podporou CUDA / NVIDIA GPU with CUDA support (testované na / tested on GTX 1070 8 GB)
- CUDA Toolkit (testované s / tested with CUDA 11.8)
- Windows 10/11

Conda prostredia pre jednotlivé algoritmy sú definované v `DP_scripts/conda_env/*.yml`. Inštalácia:
Conda environments for individual algorithms are defined in `DP_scripts/conda_env/*.yml`. Installation:

```bash
conda env create -f DP_scripts/conda_env/<algoritmus>_environment.yml
```
