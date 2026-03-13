# FFHQ-FaceFusion-10k

## SK

Testovací dataset pre diplomovú prácu zameranú na detekciu zmanipulovaných obrazov tváre. Dataset obsahuje 10 000 obrazov rozdelených rovnomerne na 5 000 autentických a 5 000 zmanipulovaných obrazov. Všetky obrazy majú rozlíšenie 1024×1024 px a sú uložené vo formáte PNG.

**Zdrojové autentické obrazy:** dataset [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) od NVIDIA (70 000 snímok tvárí, 1024×1024 px).

**Zmanipulované obrazy** boli vygenerované nástrojom [FaceFusion](https://github.com/facefusion/facefusion) v headless režime pomocou modelu deep swapper (`iperov/elon_musk_224`), vylepšovača tváre (`gpen_bfr_1024`) a obnovy výrazov (`live_portrait`).

| Kategória | Počet | Veľkosť | Pomenovanie |
|---|---|---|---|
| Autentické | 5 000 | 6,26 GB | `00005.png` – `60688.png` |
| Zmanipulované | 5 000 | 9,92 GB | `fake_00005.png` – `fake_60688.png` |

**Štruktúra:**
```
FFHQ-FaceFusion-10k/
├── real/     # Autentické FFHQ tváre
└── fake/     # Zmanipulované obrazy (prefixované fake_)
```

### Stiahnutie

Dataset je dostupný na Kaggle:

**[kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k](https://www.kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k)**

```bash
kaggle datasets download -d lukaslechovic/ffhq-facefusion-10k
```

---

## EN

A test dataset for a diploma thesis focused on detecting manipulated face images. The dataset contains 10,000 images evenly split into 5,000 authentic and 5,000 manipulated images. All images have a resolution of 1024×1024 px and are stored in PNG format.

**Source authentic images:** [FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset) dataset from NVIDIA (70,000 face images, 1024×1024 px).

**Manipulated images** were generated using the [FaceFusion](https://github.com/facefusion/facefusion) tool in headless mode with the deep swapper model (`iperov/elon_musk_224`), face enhancer (`gpen_bfr_1024`), and expression restorer (`live_portrait`).

| Category | Count | Size | Naming |
|---|---|---|---|
| Authentic | 5,000 | 6.26 GB | `00005.png` – `60688.png` |
| Manipulated | 5,000 | 9.92 GB | `fake_00005.png` – `fake_60688.png` |

**Structure:**
```
FFHQ-FaceFusion-10k/
├── real/     # Authentic FFHQ faces
└── fake/     # Manipulated images (prefixed with fake_)
```

### Download

The dataset is available on Kaggle:

**[kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k](https://www.kaggle.com/datasets/lukaslechovic/ffhq-facefusion-10k)**

```bash
kaggle datasets download -d lukaslechovic/ffhq-facefusion-10k
```
