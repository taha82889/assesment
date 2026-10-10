FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir -e . rapidocr-onnxruntime websockets
CMD ["python", "-m", "sco_ocr.main", "--run", "--state", "/data/state.json"]
