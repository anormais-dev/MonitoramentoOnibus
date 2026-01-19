# debug_crops.py
import cv2, os
from ultralytics import YOLO

YOLO_MODEL = "yolov8n.pt"
yolo = YOLO(YOLO_MODEL)
os.makedirs("debug_crops", exist_ok=True)

for img in os.listdir("detections/correct/"):
    if not img.lower().endswith((".jpg",".png")):
        continue
    im = cv2.imread("detections/correct/" + img)
    res = yolo(im)[0]
    for i, box in enumerate(res.boxes):
        cls = int(box.cls.item())
        if yolo.model.names[cls] != "bus":
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        crop = im[y1:y2, x1:x2]
        cv2.imwrite(f"debug_crops/{img}_crop{i}.jpg", crop)
