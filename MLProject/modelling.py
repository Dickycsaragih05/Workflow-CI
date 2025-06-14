import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Argparse untuk MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="emails_preprocessed.csv")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data_path)
X = df.drop(columns="Prediction")
y = df["Prediction"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("ci_retrain")
with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc:.4f}")
