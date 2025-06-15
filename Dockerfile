FROM python:3.10-slim

WORKDIR /opt/mlflow

COPY model_artifact /opt/ml/model

RUN pip install --upgrade pip && \
    pip install mlflow==3.1.0

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "5000", "--env-manager=local"]
