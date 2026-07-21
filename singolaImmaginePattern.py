import cv2
import numpy as np
import os
import time

# === PARAMETRI ===
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_DIR = os.path.join(DESKTOP_DIR, "output_single")  # salva sul Desktop
IMG_PATH = "lampa-uliczna-Ledlabs-200W-3.png"
RADIUS = 20  # dimensione punti (modifica solo questo parametro!)
CENTER_COORDS = (550, 290)  # da adattare in base all'immagine reale

# Calcola fattore di scala in base al raggio
SCALE = RADIUS / 5  # 5 = raggio di riferimento "base"

# Controllo se l'immagine esiste
if not os.path.isfile(IMG_PATH):
    raise FileNotFoundError(f"❌ File non trovato: {IMG_PATH}")

# Crea cartella di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carica immagine originale
base_image = cv2.imread(IMG_PATH)
if base_image is None:
    raise ValueError(f"❌ Impossibile leggere l'immagine '{IMG_PATH}'")

H, W = base_image.shape[:2]
cx, cy = CENTER_COORDS

# Pattern: rettangolo + centro, scalato automaticamente (senza rotazione)
pattern = np.array([
    [-30, -15],
    [30, -15],
    [30, 15],
    [-30, 15],
    [0, 0]
]) * SCALE

# Genera immagine
img = base_image.copy()

# Applica pattern direttamente al centro
points = []
for dx, dy in pattern:
    px, py = int(cx + dx), int(cy + dy)
    points.append((px, py))
    cv2.circle(img, (px, py), RADIUS, (0, 255, 0), -1)

# Bounding box YOLO
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
x_min, x_max = max(0, min(x_coords)), min(W, max(x_coords))
y_min, y_max = max(0, min(y_coords)), min(H, max(y_coords))

x_c = (x_min + x_max) / 2 / W
y_c = (y_min + y_max) / 2 / H
w = (x_max - x_min) / W
h = (y_max - y_min) / H

# Nome univoco per immagine e label (timestamp)
timestamp = int(time.time())
filename = f"sample_{timestamp}.jpg"
labelname = f"sample_{timestamp}.txt"

# Salva immagine e label
img_path = os.path.join(OUTPUT_DIR, filename)
label_path = os.path.join(OUTPUT_DIR, labelname)

cv2.imwrite(img_path, img)
with open(label_path, "w") as f:
    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

print(f"✅ Immagine salvata in: {img_path}")
print(f"✅ Label salvata in: {label_path}")

# Mostra immagine a schermo
cv2.imshow("Pattern", img)
cv2.waitKey(0)  # aspetta un tasto
cv2.destroyAllWindows()
