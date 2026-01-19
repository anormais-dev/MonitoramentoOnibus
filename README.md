Projeto focalizado na detecção de ônibus utilizando o modelo yolov8n por meio de uma camêra de segurança conectada via RTSP URL, visando salvar a hora e imagem do momento em que ele passa.

identifier.py - Algoritmo que conecta a camêra e identifica o ônibus
single_image_classifier - compara com o modelo pré-treinado para validar se o ônibus é correto ou não
training_identifier - algoritmo para treinar o identificador

dataset - pasta onde fica os exemplos que o modelo compara
detections - detecções feita por identifier
classified - pasta onde identifier salva a detecção
