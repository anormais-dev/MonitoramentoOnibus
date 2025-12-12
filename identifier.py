# camera_detection_save.py
# Requisitos: pip install ultralytics opencv-python
# Se usar GPU, instale torch com suporte CUDA antes de rodar.

import os
import time
import threading
from datetime import datetime

import cv2
from ultralytics import YOLO

# RTSP_URL = os.environ.get("RTSP_URL")

USER = "admin"
PASS = "peinha3200"
IP = "192.168.100.4"
PORT = 554
RTSP_PATH = "/cam/realmonitor?channel=1&subtype=0"  # troque se necessário
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{PORT}{RTSP_PATH}?tcp"

# ------------- CONFIG -------------
# Fonte de vídeo: pode ser RTSP (ex: rtsp://user:pass@ip:554/..) ou arquivo local "teste.mp4"

# ----------------------------------

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Threaded video capture to avoid blocking on cap.read()
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open_cap(self):
        # tenta abrir com backend padrão; se for RTSP, backend pode variar por plataforma
        try:
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(self.src)
        # tentar reduzir buffer
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _reader(self):
        backoff = 1.0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                try:
                    self.cap = self._open_cap()
                    backoff = 1.0
                except Exception:
                    self.cap = None
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 16)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # tentativa de reconexão
                try:
                    if self.cap:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue

            with self.lock:
                self.frame = frame.copy()

            # pequena pausa para liberar CPU; ajustar se precisar mais FPS
            # time.sleep(0.03) -- video
            time.sleep(0.005)

        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)

def save_detection_fullframe(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"bus_at_{ts}.jpg"
    fpath = os.path.join(DETECTIONS_DIR, fname)
    cv2.imwrite(fpath, frame)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Salvou: {fpath}")
    return fpath

def main():
    print("Inicializando modelo YOLO e video stream...")
    yolo = YOLO(YOLO_MODEL)
    yolo.overrides['verbose'] = False
    vs = VideoStream(VIDEO_SOURCE)

    frame_idx = 0
    try:
        while True:
            frame = vs.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            # preview
            if not HEADLESS:
                cv2.imshow(DISPLAY_WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_idx % DETECT_EVERY != 0:
                continue

            # inferência (pode ser lenta sem GPU)
            try:
                results = yolo(frame, imgsz=640)[0]
            except Exception as e:
                print("Erro inferência YOLO:", e)
                continue

            # percorre detecções
            if not hasattr(results, "boxes"):
                continue

            for box in results.boxes:
                # versão ultralytics: box.cls, box.conf, box.xyxy
                try:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item()) if hasattr(box, "conf") else 1.0
                except Exception:
                    # fallback se objeto diferente
                    continue

                name = yolo.model.names.get(cls, str(cls))
                if name != "bus":
                    continue
                if conf < MIN_YOLO_CONF:
                    continue

                # se chegou aqui -> detectou ônibus com confiança suficiente
                save_detection_fullframe(frame)

                # desenha bbox e label no frame (apenas visual)
                try:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = f"bus {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

            # opcional: mostrar frame atualizado com caixas
            if not HEADLESS:
                cv2.imshow(DISPLAY_WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    finally:
        vs.stop()
        cv2.destroyAllWindows()
        print("Finalizado.")

if __name__ == "__main__":
    main()
