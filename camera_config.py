import os
from dotenv import load_dotenv

load_dotenv()

def get_rtsp_url():
    protocol = os.getenv("CAMERA_PROTOCOL", "rtsp")
    user = os.getenv("CAMERA_USER")
    password = os.getenv("CAMERA_PASS")
    ip = os.getenv("CAMERA_IP")
    port = os.getenv("CAMERA_PORT", "554")
    path = os.getenv("CAMERA_PATH", "")
    transport = os.getenv("CAMERA_TRANSPORT", "tcp")

    if not all([user, password, ip]):
        raise RuntimeError("Configuração da câmera incompleta no .env")

    return f"{protocol}://{user}:{password}@{ip}:{port}{path}?{transport}"
