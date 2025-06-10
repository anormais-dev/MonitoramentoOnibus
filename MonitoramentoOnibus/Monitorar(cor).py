import cv2
import datetime

# Função auxiliar para verificar se está dentro de um horário bloqueado (ex: manhã)
def esta_em_horario_bloqueado():
    agora = datetime.datetime.now()
    hora = agora.hour
    minuto = agora.minute

    # EXEMPLO: bloquear entre 08:00 e 09:00
    if (hora == 8):
        return True
    return False

# Inicializa a captura de vídeo da câmera (0 = webcam interna, ou substitua pelo caminho RTSP/IP)
#cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.28:554/h264')

# teste
cap = cv2.VideoCapture('teste.mp4')

# Classificador de detecção de objetos (poderia ser YOLO, MobileNet ou modelo treinado para ônibus)
# Para fins de exemplo, usaremos simples detecção de contornos + cor
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona para melhorar desempenho
    resized = cv2.resize(frame, (640, 480))

    # Converte para HSV para facilitar a detecção de cores
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Define faixas de cor para azul e branco (ajustar conforme o tom do ônibus)
    azul_lower = (100, 100, 50)
    azul_upper = (130, 255, 255)

    branco_lower = (0, 0, 200)
    branco_upper = (180, 40, 255)

    # Cria máscaras para azul e branco
    mask_azul = cv2.inRange(hsv, azul_lower, azul_upper)
    mask_branco = cv2.inRange(hsv, branco_lower, branco_upper)

    # Combina as máscaras
    mask_combined = cv2.bitwise_and(mask_azul, mask_branco)

    # Encontra contornos nas áreas detectadas
    contours, _ = cv2.findContours(mask_azul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    onibus_detectado = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:  # Pode ajustar para filtrar objetos pequenos
            x, y, w, h = cv2.boundingRect(cnt)
            roi = mask_branco[y:y+h, x:x+w]
            branco_area = cv2.countNonZero(roi)

            # Se a região azul também tiver branco significativo, assume que é ônibus
            if branco_area > 600:
                onibus_detectado = True
                cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(resized, 'Onibus detectado', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Se um ônibus foi detectado
    if onibus_detectado:
        agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Comentado: bloqueia registro em horários específicos
        # if esta_em_horario_bloqueado():
        #     print(f"Onibus detectado, mas ignorado por horário bloqueado: {agora}")
        # else:
        print(f"\u2705 Onibus detectado em {agora}")
        # Aqui você pode salvar em arquivo, banco ou enviar para seu site futuramente

    # Exibe o frame com a detecção
    cv2.imshow('Deteccao de Onibus', resized)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
