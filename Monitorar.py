import cv2
import os
from ultralytics import YOLO
from datetime import datetime, timedelta
import time

# Carrega o modelo
model = YOLO("yolov8n.pt")

# Dados da câmera
rtspUrl = 'rtsp://admin:peinha3200@192.168.1.10:554/'

# Garante que a pasta "prova" exista
os.makedirs("prova", exist_ok=True)

# Função para abrir a câmera com tentativa
def abrirCamera():
    print("Conectando à câmera...")
    cap = cv2.VideoCapture(rtspUrl)
    if not cap.isOpened():
        print("❌ Falha ao abrir a câmera. Tentando novamente em 5 segundos...")
        time.sleep(5)
        return abrirCamera()
    print("✅ Conectado à câmera.")
    return cap

cap = abrirCamera()

ultimaDeteccao = datetime.min

while True:
    ret, frame = cap.read()

    # Tentativa de reconexão caso perca o sinal
    if not ret or frame is None:
        print("⚠️ Frame não recebido. Reconectando...")
        cap.release()
        cap = abrirCamera()
        continue

    try:
        results = model(frame, verbose=False)[0]
    except Exception as e:
        print(f"Erro ao rodar YOLO: {e}")
        continue

    onibusDetectado = False

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 5 and conf > 0.5:
            onibusDetectado = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Onibus {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    agora = datetime.now()
    if onibusDetectado and (agora - ultimaDeteccao).total_seconds() > 10:
        timestamp = agora.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"prova/onibus_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🚌 Ônibus detectado às {agora.strftime('%Y-%m-%d %H:%M:%S')} - imagem salva em {filename}")
        ultimaDeteccao = agora

    # Mostrar imagem ao vivo (remova se rodar sem GUI)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
