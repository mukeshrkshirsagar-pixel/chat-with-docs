FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache \
    DATA_DIR=/data \n    PORT=7860 \n    LOAD_SAMPLES_ON_START=true

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image: no network needed at runtime, faster cold starts.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY sample_docs ./sample_docs

RUN useradd --create-home --uid 1000 appuser && mkdir /data && chown appuser /data
USER appuser
VOLUME /data
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/healthz')" || exit 1

CMD exec uvicorn app.main:app_factory --factory --host 0.0.0.0 --port ${PORT}
