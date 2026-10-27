# API image for Azure Container Apps. Same as the root Dockerfile, plus the
# model exported by `python -m src.export_model`, so no MLflow server is needed.
# Build from the repository root: docker build -f deploy/azure/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY model/ model/
COPY pyproject.toml .

ENV MODEL_PATH=/app/model

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
