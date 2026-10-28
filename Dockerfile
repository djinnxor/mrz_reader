FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libopengl0 \
    libglx0 \
    libgl1-mesa-dri \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV QT_QPA_PLATFORM=offscreen
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model/face_detector /app/model/mrz_detector /app/mrz_reader/templates

COPY . .

RUN groupadd -r appuser && useradd -r -g appuser -m appuser
RUN chown -R appuser:appuser /app
USER appuser

ENV EASYOCR_MODULE_PATH=/app/.EasyOCR

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

CMD ["python", "main.py"]