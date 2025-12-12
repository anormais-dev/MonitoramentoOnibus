import cv2
import numpy as np
import joblib

# Carrega o modelo treinado
modelo = joblib.load("classificador_onibus.joblib")

# Extrai as cores médias de uma imagem (formato HSV)
def extrairCores(img, size=(64, 64)):
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img.reshape(-1, 3).mean(axis=0)

# Classifica o ROI como escolar ou municipal
def classificar(roi):
    if roi.size == 0:
        return None
    features = extrairCores(roi).reshape(1, -1)
    pred = modelo.predict(features)[0]
    return "municipal" if pred == 1 else "escolar"
