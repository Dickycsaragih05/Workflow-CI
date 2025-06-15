# MLProject/Dockerfile
FROM python:3.10-slim

WORKDIR /opt/mlflow

# Install dependencies
COPY conda.yaml .
RUN pip install --upgrade pip && \
    pip install mlflow && \
    pip install -r <(grep -A 1000 "dependencies:" conda.yaml | tail -n +2 | grep -v "dependencies:" | sed 's/- //g')

# Copy model artifact (nanti akan dibuat di step GitHub Actions)
COPY model /opt/ml/model

# Set default command
CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "5000", "--env-manager=local"]
