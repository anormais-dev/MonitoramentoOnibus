import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta

# Carrega o modelo
model = YOLO("yolov8n.pt")

# Acessa câmera
cap = cv2.VideoCapture('teste.mp4')

# Armazena o horário da última detecção registrada
ultimaDeteccao = datetime.min

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    onibusDetectado = False
 
    # Loop pelas detecções
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 5 and conf > 0.5:  # Classe 5 = ônibus
            onibusDetectado = True

            # Desenha retângulo (opcional)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Onibus {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Só imprime no terminal se se passaram 5s desde a última vez
    agora = datetime.now()
    if onibusDetectado and (agora - ultimaDeteccao).total_seconds() > 10:
        print(f"Ônibus detectado às {agora.strftime('%Y-%m-%d %H:%M:%S')}")
        ultimaDeteccao = agora

    # Mostrar imagem (remova se não quiser visualizar)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
