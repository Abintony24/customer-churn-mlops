import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "CustomerChurnModel"

# Load latest registered model
try:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        versions, key=lambda version: int(version.version)
    )

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{latest_version.version}"
    )

    print("Model loaded successfully!")

except Exception as e:
    print(f"Model loading failed: {e}")
    model = None


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": MODEL_NAME
    }


@app.post("/predict")
def predict(data: CustomerInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)

    return {
        "prediction": int(prediction[0]),
        "churn": "Yes" if prediction[0] == 1 else "No"
    }