from __future__ import annotations
import os
import cv2
import time
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

# ------------------- CONFIGURAÇÃO INICIAL ------------------------
# Carrega variáveis do arquivo .env
load_dotenv() 

# URL da câmera RTSP definida no .env
RTSP_URL = os.getenv("RTSP_URL") 

# Parâmetros fixos do sistema
MODEL_PATH    = "yolov8s.pt" # Caminho do modelo YOLO
SAVE_DIR      = Path("prova") # Pasta onde salvará imagens
ROI           = (600, 200, 1350, 650) # Região da imagem a ser analisada (x1, y1, x2, y2)
CONF_TH       = 0.55 # Limite mínimo de confiança para considerar uma detecção
AREA_MIN      = 50_000 # Área mínima da detecção (evita vans e veículos pequenos)
ASPECT_RANGE  = (1.8, 4.5) # Faixa aceitável da razão largura/altura do objeto
COOLDOWN_SEC  = 10 # Intervalo mínimo entre salvamentos consecutivos

# --------------------- CONFIGURAÇÃO DE LOG ------------------------

logging.basicConfig(
    level=logging.INFO, # Nível mínimo de log: INFO
    format="%(asctime)s [%(levelname)s] %(message)s",  # Formato da mensagem
    handlers=[
        logging.FileHandler("bus_watcher.log", encoding="utf-8"),  # Salva em arquivo
        logging.StreamHandler() # Mostra no terminal
    ]
)

# --------------------- INICIALIZAÇÃO DO MODELO --------------------

SAVE_DIR.mkdir(exist_ok=True) # Cria pasta de saída, se não existir
model = YOLO(MODEL_PATH).to("cpu") # Carrega modelo YOLO na CPU (use .to("cuda") para GPU)

# --------------------- FUNÇÃO: Conectar à câmera ------------------
def open_camera() -> cv2.VideoCapture:
    backoff = 5
    while True:
        logging.info("Conectando à câmera…")
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if cap.isOpened():
            logging.info("Conectado.")
            return cap
        logging.warning("Falha. Tentando novamente em %ss…", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)  # Aumenta o tempo até 60 segundos

# ------------------ FUNÇÃO: Detectar ônibus -----------------------
def detect_buses(frame):
    x1, y1, x2, y2 = ROI
    roi_img = frame[y1:y2, x1:x2]  # Recorta a região de interesse (ROI)

    try:
        preds = model(roi_img, verbose=False)[0]  # Executa a predição
    except Exception as e:
        logging.error("YOLO error: %s", e)
        return []

    buses = []
    for b in preds.boxes:
        if int(b.cls[0]) != 5 or float(b.conf[0]) < CONF_TH:  # Classe 5 = ônibus no COCO
            continue

        bx1, by1, bx2, by2 = map(int, b.xyxy[0])  # Coordenadas do bounding box
        w, h = bx2 - bx1, by2 - by1 # Calcula largura e altura
        area = w * h
        ratio = w / h if h else 0 # Razão largura/altura

        # Verifica se área e proporção estão dentro dos critérios
        if area >= AREA_MIN and ASPECT_RANGE[0] <= ratio <= ASPECT_RANGE[1]:
            # Ajusta coordenadas para o frame completo
            buses.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, float(b.conf[0])))

    return buses

# ------------------- LOOP PRINCIPAL -------------------------------

cap = open_camera() # Conecta à câmera
last_saved = datetime.min # Marca inicial para cooldown

while True:
    ok, frame = cap.read() # Lê um frame da câmera

    if not ok: # Caso falhe, reconecta
        logging.warning("Frame não recebido. Reabrindo RTSP.")
        cap.release()
        cap = open_camera()
        continue

    buses = detect_buses(frame)  # Detecta ônibus no frame atual
    now = datetime.now()

    # Salva imagem se houver detecção e já passou o tempo mínimo (cooldown)
    if buses and (now - last_saved).total_seconds() >= COOLDOWN_SEC:
        bx1, by1, bx2, by2, conf = max(buses, key=lambda b: b[2] - b[0])  # Pega o maior ônibus detectado

        # Desenha retângulo e texto no frame
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, f"Ônibus {conf:.2f}", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Salva o recorte do ônibus detectado
        crop = frame[by1:by2, bx1:bx2]
        filename = SAVE_DIR / f"bus_{now:%Y-%m-%d_%H-%M-%S}.jpg"
        cv2.imwrite(str(filename), crop)

        logging.info("🚌 Detectado (%0.2f). Salvo %s", conf, filename)
        last_saved = now

    # Exibe imagem em tempo real (remover em modo servidor/headless)
    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para sair
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
