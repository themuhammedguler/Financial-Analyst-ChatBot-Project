# ==============================================================
# BIST Finansal Analist Asistanı - Production Dockerfile
# Multi-platform & Cloud Ready (Python 3.11 Slim)
# ==============================================================

FROM python:3.11-slim

# Sistem bağımlılıkları ve curl/git optimizasyonu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python ortam değişkenleri
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Bağımlılıkları kopyala ve kur (Docker katman önbelleği için önce bu adım)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını ve kaynakları kopyala
COPY . /app

# Kalıcı dizinlerin varlığını garanti altına al
RUN mkdir -p /app/reports /app/data /app/vectorstore/db_faiss

# Portları aç (Streamlit: 8501, FastAPI: 8000)
EXPOSE 8501 8000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Varsayılan başlangıç komutu (Streamlit UI)
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
