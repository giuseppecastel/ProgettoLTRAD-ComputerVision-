import cv2
import numpy as np
import random
from ultralytics import YOLO
from PIL import Image
import os

# === 📷 INPUT PERSONALIZZABILI ===
image_path = "immagine_test.jpg"   # 🔁 La tua immagine reale (da testare)
output_path = "immagine_test_pattern.jpg"  # 🔁 Dove salvare l'immagine con il pattern
cx, cy = 320, 320  # 🔁 Dove vuoi centrare il pattern (coordinata x, y)

# === ⚙️ PARAMETRI PATTERN ===
IMG_SIZE = 640       # Ridimensionamento immagine (opzionale ma consigliato)
RADIUS = 5           # Raggio dei pallini
pattern = np.array([
    [-30, -15],
    [30, -15],
    [30, 15],
    [-30, 15],
    [0, 0]
])

# === 1. Carica immagine reale
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"❌ Immagine '{image_path}' non trovata!")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize coerente col modello YOLO

# === 2. Rotazione casuale pattern (simula caso realistico)
angle = random.uniform(-30, 30)
rot_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
rotated = np.dot(pattern, rot_matrix[:, :2].T)

# === 3. Disegna il pattern
for dx, dy in rotated:
    px, py = int(cx + dx), int(cy + dy)
    cv2.circle(img, (px, py), RADIUS, (0, 255, 0), -1)

# === 4. Salva immagine pronta per il test
cv2.imwrite(output_path, img)
print(f"✅ Immagine con pattern salvata in: {output_path}")

# === 5. Carica modello YOLO e fai il test
model = YOLO("runs/detect/model_pattern_lampione/weights/best.pt")

results = model(output_path, save=True)

# === 6. Stampa risultati
for r in results:
    print("\n🧠 RISULTATI RILEVATI:")
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"- Oggetto {i+1}: Classe {cls_id}, Confidenza: {conf:.2f}, Box: {xyxy}")

# === 7. Mostra immagine con predizione
predicted_path = "runs/detect/predict/immagine_test_pattern.jpg"
if os.path.exists(predicted_path):
    Image.open(predicted_path).show()
else:
    print("⚠️ Immagine con predizioni non trovata")
