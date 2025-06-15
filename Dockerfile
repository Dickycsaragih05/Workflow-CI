FROM python:3.10-slim

WORKDIR /opt/mlflow

# Copy model dan file environment
COPY model /opt/ml/model

# Install dependencies
RUN pip install --upgrade pip && \
    pip install mlflow==3.1.0 && \
    python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, env_manager='local')"

# Jalankan MLflow model serving
ENTRYPOINT ["mlflow"]
CMD ["models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "5000", "--env-manager=local"]
