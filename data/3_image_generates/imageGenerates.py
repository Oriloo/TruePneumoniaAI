import os
import cv2
import numpy as np
import random
import shutil

# === CONFIGURATION ===
TARGET_PER_CLASS = 5000
INPUT_DIR = "../2_image_resize/outputs"
OUTPUT_DIR = "outputs"

# === TRANSFORMATIONS ===

def apply_blur(img):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_brightness(img):
    value = random.randint(-40, 40)
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)

def apply_mirror_horizontal(img):
    return cv2.flip(img, 1)

def apply_rotation(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=0)

def apply_clahe(img):
    clip = random.uniform(1.0, 4.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_zoom_crop(img):
    h, w = img.shape[:2]
    scale = random.uniform(1.05, 1.20)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    x_start = random.randint(0, new_w - w)
    y_start = random.randint(0, new_h - h)
    return resized[y_start:y_start + h, x_start:x_start + w]

def apply_translation(img):
    h, w = img.shape[:2]
    tx = random.randint(-30, 30)
    ty = random.randint(-30, 30)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h), borderValue=0)

def apply_gaussian_noise(img):
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

def apply_gamma(img):
    gamma = random.uniform(0.6, 1.5)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

ALL_TRANSFORMS = [
    apply_blur,
    apply_brightness,
    apply_mirror_horizontal,
    apply_rotation,
    apply_clahe,
    apply_zoom_crop,
    apply_translation,
    apply_gaussian_noise,
    apply_gamma,
]

def augment_image(img):
    transforms = random.sample(ALL_TRANSFORMS, 3)
    for t in transforms:
        img = t(img)
    return img

# === CLASSIFICATION DES IMAGES PAR CLASSE ===

def classify_image(filename):
    if "bacteria" in filename:
        return "bacteria"
    elif "virus" in filename:
        return "virus"
    else:
        return "normal"

# === MAIN ===

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    classes = {"normal": [], "bacteria": [], "virus": []}
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".jpg"):
            classes[classify_image(filename)].append(filename)

    print("=== Images sources ===")
    for cls, files in classes.items():
        print(f"  {cls}: {len(files)}")
    print(f"  Cible par classe: {TARGET_PER_CLASS}")
    print()

    global_id = 0
    for cls, files in classes.items():
        nb_originals = len(files)
        nb_to_generate = max(0, TARGET_PER_CLASS - nb_originals)

        print(f"[{cls}] {nb_originals} originales, {nb_to_generate} à générer...")

        for filename in files:
            src = os.path.join(INPUT_DIR, filename)
            dst = os.path.join(OUTPUT_DIR, f"{cls}-{global_id}.jpg")
            shutil.copy2(src, dst)
            global_id += 1

        for i in range(nb_to_generate):
            src_filename = random.choice(files)
            src_path = os.path.join(INPUT_DIR, src_filename)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            augmented = augment_image(img)
            dst = os.path.join(OUTPUT_DIR, f"{cls}-{global_id}.jpg")
            cv2.imwrite(dst, augmented)
            global_id += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{nb_to_generate} générées")

        print(f"  Terminé ! Total {cls}: {nb_originals + nb_to_generate}")
        print()

    total = 0
    for cls in classes:
        count = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith(cls + "-")])
        total += count
        print(f"Nombre d'images {cls}: {count}")
    print(f"Nombre total d'images: {total}")

if __name__ == "__main__":
    main()
