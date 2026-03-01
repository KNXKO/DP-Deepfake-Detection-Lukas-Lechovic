# Generovanie vizuálnych informácií a rozpoznávanie generovaného či zmanipulovaného obsahu

**Autor:** Lukáš Lechovič
**Téma:** Generovanie vizuálnych informácií a rozpoznávanie generovaného či zmanipulovaného obsahu
**Vedúci práce:** PaedDr. Miroslav Ölvecký, PhD.

Repozitár obsahuje praktickú časť diplomovej práce. Zahŕňa automatizačný skript na generovanie testovacieho datasetu pomocou nástroja FaceFusion a implementáciu štyroch detekčných algoritmov testovaných na vlastnom datasete FFHQ-FaceFusion-10k.

---

## Štruktúra projektu

```
DP-PraktickaCast_Lukas-Lechovic/
│
├── DP_Lukas_Lechovic_FaceFusion/
│   └── Script_AutomatedGenerateImages/
│       └── process_male_faces.bat        # Dávkový skript na generovanie zmanipulovaných obrazov
│
└── DP_Lukas_Lechovic_DetectionAlgorithms/
    ├── AIDE/                             # Algoritmus AIDE
    │   └── AIDE/DP_scripts/             # Vlastné testovacie skripty
    ├── CLIP/                             # Algoritmus CLIP-ViT (UniversalFakeDetect)
    │   └── CLIP/DP_scripts/
    ├── CNN/                              # Algoritmus CNNDetection
    │   └── CNNDetection/DP_scripts/
    └── DeepFakeBench/                    # Algoritmus Xception (cez DeepFakeBench)
        └── DP_scripts/
```

Dataset nie je súčasťou repozitára kvôli veľkosti (16,1 GB). Postup stiahnutia je popísaný nižšie.

---

## Testovací dataset – FFHQ-FaceFusion-10k

Dataset obsahuje 10 000 obrazov rozdelených rovnomerne na 5 000 autentických a 5 000 zmanipulovaných obrazov. Všetky obrazy majú rozlíšenie 1024×1024 px a sú uložené vo formáte PNG.

| Kategória | Počet | Veľkosť | Pomenovanie |
|---|---|---|---|
| Autentické | 5 000 | 6,26 GB | `00005.png` – `60688.png` |
| Zmanipulované | 5 000 | 9,92 GB | `fake_00005.png` – `fake_60688.png` |

**Zdrojové autentické obrazy:** dataset [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) od NVIDIA (70 000 snímok tvárí, 1024×1024 px).

**Štruktúra datasetu na disku:**
```
FFHQ-FaceFusion-10k/
├── real/     # Autentické FFHQ tváre
└── fake/     # Zmanipulované obrazy (prefixované fake_)
```

### Stiahnutie datasetu

Dataset FFHQ-FaceFusion-10k je dostupný na Kaggle:

**[kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k](https://www.kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k)**

```bash
# Stiahnutie cez Kaggle CLI (vyžaduje kaggle API token)
kaggle datasets download -d lukaslechovic/ffhq-facefusion-10k
```

---

## Generovanie zmanipulovaných obrazov

Zmanipulované obrazy boli vygenerované automatizovaným dávkovým skriptom `process_male_faces.bat` pomocou nástroja FaceFusion v headless režime (bez grafického rozhrania).

**Požiadavky:**
- Nainštalovaná aplikácia [Pinokio](https://pinokio.computer/) s FaceFusion

**Parametre použité pri generovaní:**

| Komponent | Model | Funkcia |
|---|---|---|
| Výmena tváre (Deep Swapper) | `iperov/elon_musk_224` | Výmena tváre pomocou hlbokého učenia |
| Vylepšovač tváre (Face Enhancer) | `gpen_bfr_1024` | Vylepšenie kvality a odstránenie artefaktov |
| Obnova výrazov (Expression Restorer) | `live_portrait` | Zachovanie pôvodných výrazov tváre |
| Výstupné rozlíšenie | 1024×1024 px | – |
| Kvalita JPEG | 90 % | – |

```bash
cd DP_Lukas_Lechovic_FaceFusion/Script_AutomatedGenerateImages
process_male_faces.bat
```

Generovanie jedného obrazu trvalo v priemere ~13 sekúnd (GPU GTX 1070 8 GB). Celkový čas generovania 5 000 obrazov bol ~18 hodín.

---

## Detekčné algoritmy

Každý algoritmus má v priečinku `DP_scripts/` vlastný skript `evaluate_detector.py` a dávkový spúšťací skript `run_evaluation.bat`. Conda prostredia sú definované v `DP_scripts/conda_env/*.yml`.

Spoločné parametre testovania:
- Prahová hodnota klasifikácie: **0,5**
- Dataset: **10 000 obrazov** (5 000 real + 5 000 fake)
- Výstup: JSON súbor s metrikami (`DP_scripts/vysledky/metrics.json`)

---

### Xception (cez DeepFakeBench)

**Repozitár:** [github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
**Conda prostredie:** Python 3.8, PyTorch ≥1.13.0, CUDA

```bash
conda activate deepfakebench

# Krok 1: Príprava JSON konfigurácie datasetu
cd DP_Lukas_Lechovic_DetectionAlgorithms/DeepFakeBench/DP_scripts
python prepare_dataset.py --dataset_path C:/FFHQ-FaceFusion-10k --dataset_name MyDataset

# Krok 2: Spustenie hodnotenia
run_evaluation.bat
# alebo manuálne:
python evaluate_detector.py \
    --detector_path ../training/config/detector/xception.yaml \
    --weights_path ../training/weights/xception_best.pth \
    --test_dataset MyDataset
```

Vstupné rozlíšenie pre model: 299×299 px. Čas spracovania ~12 hodín.

---

### CLIP-ViT (UniversalFakeDetect)

**Repozitár:** [github.com/WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
**Conda prostredie:** Python 3.10, PyTorch ≥1.13.0, CUDA

```bash
conda activate clip

cd DP_Lukas_Lechovic_DetectionAlgorithms/CLIP/CLIP/DP_scripts
run_evaluation.bat
# alebo manuálne:
python evaluate_detector.py \
    --real_path C:/FFHQ-FaceFusion-10k/real \
    --fake_path C:/FFHQ-FaceFusion-10k/fake \
    --checkpoint ../pretrained_weights/fc_weights.pth \
    --output_dir vysledky
```

Vstupné rozlíšenie pre model: 224×224 px (resize na 256, center crop). Čas spracovania ~11 hodín.

---

### AIDE

**Repozitár:** [github.com/shilinyan99/AIDE](https://github.com/shilinyan99/AIDE)
**Conda prostredie:** Python 3.10, PyTorch 2.0.1, CUDA 11.8

Algoritmus vyžaduje tri checkpointy:
- `checkpoints/GenImage_train.pth` – hlavný AIDE checkpoint
- `checkpoints/resnet50.pth` – ResNet-50
- `checkpoints/open_clip_pytorch_model.bin` – ConvNeXt (OpenCLIP)

```bash
conda activate aide

cd DP_Lukas_Lechovic_DetectionAlgorithms/AIDE/AIDE/DP_scripts
run_evaluation.bat
# alebo manuálne:
python evaluate_detector.py \
    --eval_data_path C:/FFHQ-FaceFusion-10k \
    --checkpoint ../AIDE/checkpoints/GenImage_train.pth \
    --resnet_path ../AIDE/checkpoints/resnet50.pth \
    --convnext_path ../AIDE/checkpoints/open_clip_pytorch_model.bin
```

Štruktúra datasetu pre AIDE: priečinky `0_real/` a `1_fake/` (skript konvertuje automaticky). Čas spracovania ~11 hodín.

---

### CNNDetection

**Repozitár:** [github.com/PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection)
**Conda prostredie:** Python 3.10, PyTorch ≥1.2.0, CUDA

Predtrénované váhy (`blur_jpg_prob0.5.pth`) nie sú súčasťou repozitára a je ich potrebné stiahnuť manuálne podľa pokynov v `readme.md` projektu CNNDetection.

```bash
conda activate cnn

cd DP_Lukas_Lechovic_DetectionAlgorithms/CNN/CNNDetection/DP_scripts
run_evaluation.bat
# alebo manuálne:
python evaluate_detector.py \
    -d C:/FFHQ-FaceFusion-10k \
    -m ../weights/blur_jpg_prob0.5.pth \
    --threshold 0.5
```

Model: ResNet-50 trénovaný na ProGAN obrazoch. Vstupné rozlíšenie pre model: 224×224 px. Čas spracovania ~14 hodín.

---

## Výstup hodnotenia

Každý algoritmus ukladá výsledky do `DP_scripts/vysledky/`:

| Súbor | Obsah |
|---|---|
| `metrics.json` | Celková presnosť, presnosť, citlivosť, F1-skóre, špecifickosť, AUC-ROC, matica zámen |
| `confusion_matrix.png` | Vizualizácia matice zámen |
| `roc_curve.png` | ROC krivka |
| `precision_recall_curve.png` | Precision-Recall krivka |

---

## Požiadavky

- Anaconda / Miniconda
- NVIDIA GPU s podporou CUDA (testované na GTX 1070 8 GB)
- CUDA Toolkit (testované s CUDA 11.8)
- Windows 10/11

Conda prostredia pre jednotlivé algoritmy sú definované v `DP_scripts/conda_env/*.yml`. Inštalácia:

```bash
conda env create -f DP_scripts/conda_env/<algoritmus>_environment.yml
```