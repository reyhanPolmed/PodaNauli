FROM node:22-alpine AS frontend-build

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/index.html ./index.html
COPY frontend/tsconfig.json frontend/tsconfig.app.json frontend/tsconfig.node.json ./
COPY frontend/vite.config.ts ./vite.config.ts
COPY frontend/src/ ./src/

ENV VITE_API_BASE_URL=/api/v1

RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY api/requirements.txt /tmp/requirements.txt
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt \
    && useradd --create-home --uid 10001 podanauli

COPY api/ /app/api/
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/model_registry.json /app/models/model_registry.json
COPY models/sentiment_champion.joblib /app/models/sentiment_champion.joblib
COPY models/complaint_detector.joblib /app/models/complaint_detector.joblib
COPY models/aspect_champion.joblib /app/models/aspect_champion.joblib
COPY models/aspect_multilabel_binarizer.joblib /app/models/aspect_multilabel_binarizer.joblib
COPY models/aspect_metadata.json /app/models/aspect_metadata.json
COPY demo/demo_metrics.json /app/demo/demo_metrics.json
COPY demo/demo_data_quality.json /app/demo/demo_data_quality.json
COPY data/processed/places_master.parquet /app/data/processed/places_master.parquet
COPY data/processed/reviews_clean.parquet /app/data/processed/reviews_clean.parquet
COPY data/processed/review_aspect_sentiment.parquet /app/data/processed/review_aspect_sentiment.parquet
COPY data/processed/place_clusters.parquet /app/data/processed/place_clusters.parquet
COPY outputs/predictions/service_gap_rankings.csv /app/outputs/predictions/service_gap_rankings.csv
COPY outputs/maps/place_clusters.geojson /app/outputs/maps/place_clusters.geojson
COPY outputs/reports/project_readiness.json /app/outputs/reports/project_readiness.json
COPY outputs/reports/data_quality_summary.csv /app/outputs/reports/data_quality_summary.csv
COPY --from=frontend-build /build/dist /app/frontend/dist

RUN mkdir -p /app/data/runtime \
    && chown -R podanauli:podanauli /app/data/runtime

USER podanauli

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
