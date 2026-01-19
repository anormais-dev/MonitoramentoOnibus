import cv2
import os
import time
import queue
from threading import Thread
from datetime import datetime
from ultralytics import YOLO
from single_image_classifier import classify_single
from camera_config import get_rtsp_url


# ================== CONFIG ==================

VIDEO_SOURCE = get_rtsp_url()  # webcam | ou rtsp://...
YOLO_MODEL = "yolov8n.pt"

DETECTIONS_DIR = "detections"
CLASSIFIED_CORRECT = "classified/correct"
CLASSIFIED_FALSE = "classified/false"
CLASSIFIED_UNKNOWN = "classified/unknown"

CONFIDENCE_THRESHOLD = 0.6
QUEUE_MAXSIZE = 300

# ============================================

os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(CLASSIFIED_CORRECT, exist_ok=True)
os.makedirs(CLASSIFIED_FALSE, exist_ok=True)

classification_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

# ================== CLASSIFIER THREAD ==================

def classification_worker():
    while True:
        img_path = classification_queue.get()
        if img_path is None:
            break

        try:
            result = classify_single(img_path)

            if (
                not result or
                "predicted" not in result or
                result["predicted"].lower() not in ("correct", "false")
            ):
                raise ValueError("Resultado inválido do classificador")

            pred = result["predicted"].lower()

            if pred == "correct":
                dest_dir = CLASSIFIED_CORRECT
            else:
                dest_dir = CLASSIFIED_FALSE

        except Exception as e:
            # ⬇️ erro real → unknown
            print("[CLASSIFIER ERROR]", e)
            dest_dir = CLASSIFIED_UNKNOWN

        # cria a pasta SOMENTE quando necessário
        os.makedirs(dest_dir, exist_ok=True)

        try:
            dest = os.path.join(dest_dir, os.path.basename(img_path))
            os.replace(img_path, dest)
            print(f"[CLASSIFIER] {os.path.basename(img_path)} → {os.path.basename(dest_dir)}")
        except Exception as e:
            print("[MOVE ERROR]", e)

        classification_queue.task_done()

# ================== UTILS ==================

def save_detection(frame):
    ts = datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S")
    filename = f"bus_{ts}.jpg"
    path = os.path.join(DETECTIONS_DIR, filename)
    cv2.imwrite(path, frame)
    return path

# ================== MAIN ==================

def main():
    print("[INIT] Loading YOLO...")
    model = YOLO(YOLO_MODEL)

    print("[INIT] Opening video source...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    classifier_thread = Thread(target=classification_worker, daemon=True)
    classifier_thread.start()

    print("[RUNNING] Press CTRL+C to stop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                label = model.names.get(cls)

                if label == "bus" and conf >= CONFIDENCE_THRESHOLD:
                    img_path = save_detection(frame)

                    try:
                        classification_queue.put_nowait(img_path)
                    except queue.Full:
                        print("[WARN] Classification queue full. Dropping image.")

            # opcional: remover se rodar headless
            cv2.imshow("Detector", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        classification_queue.put(None)
        print("[SHUTDOWN] Waiting classifier thread...")
        classifier_thread.join()

# ================== ENTRY ==================

if __name__ == "__main__":
    main()
