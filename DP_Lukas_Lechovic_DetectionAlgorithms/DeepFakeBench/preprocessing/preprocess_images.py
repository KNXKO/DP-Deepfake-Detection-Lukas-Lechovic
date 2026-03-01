import os
import sys
import time
import cv2
import dlib
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans


def create_logger(log_path):
    """
    Vytvorí logger objekt a uloží všetky správy do súboru.
    """
    # Vytvor priečinok pre logy ak neexistuje
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_keypts(image, face, predictor, face_detector):
    # Detekuje kľúčové body tváre
    shape = predictor(image, face)

    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """
        Zarovná a orezáva tvár podľa kľúčových bodov
        """
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb, 1)
    if len(faces):
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = get_keypts(rgb, face, predictor, face_detector)
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    else:
        return None, None, None


def image_manipulate(image_path: Path, dataset_path: Path, logger) -> None:
    """
    Spracuje jeden obrázok - detekuje a orezáva tvár a ukladá výsledky.
    """
    # Definuj face detector a predictor modely
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'

    if not os.path.exists(predictor_path):
        logger.error(f"Predictor cesta neexistuje: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)

    try:
        # Načítaj obrázok
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Nepodarilo sa načítať obrázok: {image_path}")
            return

        # Extrahuj zarovnanú a orezanú tvár
        cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, face_predictor, image)

        # Skontroluj, či bola tvár detekovaná a orezaná
        if cropped_face is None:
            logger.warning(f"Žiadne tváre v obrázku: {image_path}")
            return

        # Skontroluj, či boli kľúčové body detekované
        if landmarks is None:
            logger.warning(f"Žiadne kľúčové body v obrázku: {image_path}")
            return

        # Ulož orezanú tvár a kľúčové body
        save_path_ = dataset_path / 'frames' / image_path.stem
        save_path_.mkdir(parents=True, exist_ok=True)

        # Ulož orezanú tvár
        image_save_path = save_path_ / "000.png"  # Používam 000 pre jednotlivé obrázky
        if not image_save_path.is_file():
            cv2.imwrite(str(image_save_path), cropped_face)

        # Ulož kľúčové body
        land_path = dataset_path / 'landmarks' / image_path.stem / "000.npy"
        os.makedirs(os.path.dirname(land_path), exist_ok=True)
        np.save(str(land_path), landmarks)

    except Exception as e:
        logger.error(f"Chyba pri spracovaní obrázka {image_path}: {e}")


def preprocess(dataset_path, logger):
    """Spracuje všetky obrázky v datasete"""
    # Definuj cesty k obrázkom v datasete
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images_path_list = []

    for ext in image_extensions:
        images_path_list.extend(sorted([Path(p) for p in glob.glob(os.path.join(dataset_path, '**', ext), recursive=True)]))

    if len(images_path_list) == 0:
        logger.error(f"Žiadne obrázky neboli nájdené v {dataset_path}")
        sys.exit()
    logger.info(f"{len(images_path_list)} obrázkov nájdených v {dataset_path}")

    # Spusti timer
    start_time = time.monotonic()

    # Definuj počet procesov podľa možností CPU
    num_processes = max(1, os.cpu_count() // 2)  # Použij polovicu jadier

    # Použij multiprocessing na paralelné spracovanie obrázkov
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for image_path in images_path_list:
            # Vytvor future pre každý obrázok a odošli ho na spracovanie
            futures.append(
                executor.submit(
                    image_manipulate,
                    image_path,
                    dataset_path,
                    logger
                )
            )

        # Počkaj, kým sa všetky futures dokončia a zaloguj chyby
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(images_path_list)):
            logger.info(f"Aktuálny čas: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                future.result()
            except Exception as e:
                logger.error(f"Chyba pri spracovaní obrázka: {e}")

        # Ukonči timer
        end_time = time.monotonic()
        duration_minutes = (end_time - start_time) / 60
        logger.info(f"Celkový čas: {duration_minutes:.2f} minút")


if __name__ == '__main__':
    # Načítaj parametre z config.yaml
    yaml_path = './config.yaml'
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("Chyba pri parsovaní YAML súboru:", e)

    # Získaj parametre
    dataset_name = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']

    # Použij dataset_name a dataset_root_path na získanie dataset_path
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    # Vytvor logger
    log_path = f'./logs/rgb/{dataset_name}.log'
    logger = create_logger(log_path)

    # Definuj cesty k datasetom podľa vstupných argumentov
    if dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    else:
        logger.error(f"Dataset {dataset_name} nie je rozpoznaný pre spracovanie obrázkov")
        sys.exit()

    # Skontroluj, či cesta k datasetu existuje
    if not Path(dataset_path).exists():
        logger.error(f"Cesta k datasetu neexistuje: {dataset_path}")
        sys.exit()

    if 'sub_dataset_paths' in globals() and len(sub_dataset_paths) != 0:
        # Skontroluj, či sub_dataset cesty existujú
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset cesta neexistuje: {sub_dataset_path}")
                sys.exit()

        # Spracuj každý sub_dataset
        for sub_dataset_path in sub_dataset_paths:
            logger.info(f"Spracovávam: {sub_dataset_path}")
            preprocess(sub_dataset_path, logger)
    else:
        logger.error(f"Sub Dataset cesta neexistuje: {sub_dataset_paths}")
        sys.exit()

    logger.info("Orezávanie tváří dokončené!")