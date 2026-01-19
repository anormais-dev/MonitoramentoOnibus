# arquivo para croppar imagens

import os
import cv2
from ultralytics import YOLO

# INPUT_DIR = "dataset/training/false"    # onde estão suas imagens originais
# OUTPUT_DIR = "dataset/training/false_cropped"     # crops para treinar
INPUT_DIR = "classified"    # onde estão suas imagens originais
OUTPUT_DIR = "classified/false_cropped"     # crops para treinar
YOLO_MODEL = "yolov8n.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

yolo = YOLO(YOLO_MODEL)

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)

    res = yolo(img)[0]
    for box in res.boxes:
        cls = int(box.cls.item())
        name = yolo.model.names.get(cls)

        if name == "bus":
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            out = os.path.join(OUTPUT_DIR, file)
            cv2.imwrite(out, crop)
            print("Crop salvo:", out)
