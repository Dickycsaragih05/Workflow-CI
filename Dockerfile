# Gunakan Python image ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin isi folder model
COPY MLProject/outputs/model.pkl /opt/ml/model/model.pkl

# Install MLflow dan dependencies
RUN pip install mlflow==3.1.0 scikit-learn pandas joblib

# Jalankan MLflow model serving
ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "5000", "--env-manager=local"]
