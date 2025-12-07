FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/input_videos /app/outputs

#Ajustar video de entrada para aplicar módulos y nombre del archivo de salida
CMD ["python", "main.py", "--video", "/app/input_videos/video_niebla_dia.mp4", "--headless", "--out", "/app/outputs/output_result_dayfog.mp4"]
