FROM python:3.11-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --disable-pip-version-check \
    -r requirements.txt

COPY camera_config.py .
COPY identifier.py .
COPY single_image_classifier.py .
COPY training_identifier.py .
COPY crops.py .
COPY .env.example .env

COPY yolov8n.pt .
COPY bus_classifier_resnet18.pth .
COPY dataset_index.faiss .
COPY dataset_meta.npy .
COPY renders_index.faiss .
COPY renders_meta.npy .

RUN mkdir -p detections classified/correct classified/false classified/unknown

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "identifier.py"]
