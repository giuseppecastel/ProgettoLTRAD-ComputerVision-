import cv2
import numpy as np
import os
import random
import shutil
from ultralytics import YOLO

# === PARAMETRI ===
NUM_IMAGES = 200
IMG_SIZE = 640
TRAIN_SPLIT = 0.8
RADIUS = 5
PATTERN_NAME = "pallini_lampione"
CENTER_COORDS = (320, 320)  # nuovo centro centrato sulla copertura

base_dir = "dataset"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")

# Reset cartelle
for d in [image_dir, label_dir]:
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "train"))
    os.makedirs(os.path.join(d, "val"))

# Carica immagine lampione aggiornata
base_image = cv2.imread("lampa-uliczna-Ledlabs-200W-3.png")
if base_image is None:
    raise FileNotFoundError("❌ Immagine 'lampa-uliczna-Ledlabs-200W-3.png' non trovata!")

# Ridimensiona a 640x640
base_image = cv2.resize(base_image, (IMG_SIZE, IMG_SIZE))
cx, cy = CENTER_COORDS

# Pattern: rettangolo + centro
pattern = np.array([
    [-30, -15],
    [30, -15],
    [30, 15],
    [-30, 15],
    [0, 0]
])

# === Genera immagini ===
for i in range(NUM_IMAGES):
    img = base_image.copy()

    # Rotazione casuale
    angle = random.uniform(-30, 30)
    rot_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated = np.dot(pattern, rot_matrix[:, :2].T)

    # Applica pattern
    points = []
    for dx, dy in rotated:
        px, py = int(cx + dx), int(cy + dy)
        points.append((px, py))
        cv2.circle(img, (px, py), RADIUS, (0, 255, 0), -1)

    # Bounding box
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = max(0, min(x_coords)), min(IMG_SIZE, max(x_coords))
    y_min, y_max = max(0, min(y_coords)), min(IMG_SIZE, max(y_coords))

    x_c = (x_min + x_max) / 2 / IMG_SIZE
    y_c = (y_min + y_max) / 2 / IMG_SIZE
    w = (x_max - x_min) / IMG_SIZE
    h = (y_max - y_min) / IMG_SIZE

    # Train/val
    subset = "train" if i < NUM_IMAGES * TRAIN_SPLIT else "val"
    filename = f"{i:04d}.jpg"
    labelname = f"{i:04d}.txt"

    # Salva immagine
    cv2.imwrite(os.path.join(image_dir, subset, filename), img)

    # Salva label YOLO
    with open(os.path.join(label_dir, subset, labelname), "w") as f:
        f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# === YAML per YOLOv8 ===
yaml_path = os.path.join(base_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

names:
  0: {PATTERN_NAME}
""")

print("✅ Dataset pronto!")

# === Addestramento YOLOv8 ===
print("🚀 Avvio addestramento...")

model = YOLO("yolov8n.pt")
model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name="model_pattern_lampione"
)
