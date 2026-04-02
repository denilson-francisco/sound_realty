FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy model artifacts, data, and application code
# Build context is the project root (set in docker-compose.yml)
COPY model ./model
COPY data/zipcode_demographics.csv ./data/zipcode_demographics.csv
COPY api/main.py .

ENV MODEL_PATH=model/model.pkl
ENV FEATURES_PATH=model/model_features.json
ENV PERCENTILES_PATH=model/model_percentiles.json
ENV DEMOGRAPHICS_PATH=data/zipcode_demographics.csv
ENV DEFAULTS_PATH=model/model_defaults.json
ENV MODEL_NAME=sound-realty-price-predictor
ARG MODEL_VERSION=1.0.0
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MODEL_STAGE=development

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
