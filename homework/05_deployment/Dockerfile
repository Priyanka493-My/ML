FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, model, and metadata/defaults
COPY predict.py .
COPY model.py .
COPY pipeline.bin .
COPY feature_defaults.json .
COPY model_metadata.json .

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]