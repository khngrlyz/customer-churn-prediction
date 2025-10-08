"""
FastAPI web service for customer churn prediction.
Accepts customer data in JSON format and returns churn probability.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from data_preprocessing import ChurnDataPreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability based on customer attributes",
    version="1.0.0"
)

# Load model and preprocessor at startup
try:
    model = joblib.load('models/churn_model.joblib')
    preprocessor = ChurnDataPreprocessor()
    preprocessor.load('models/preprocessor.joblib')
    print("Model and preprocessor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None


# Input schema
class CustomerData(BaseModel):
    """Customer data input schema."""
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=75.50)
    TotalCharges: float = Field(..., ge=0, example=900.00)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.50,
                "TotalCharges": 900.00
            }
        }


# Output schema
class PredictionResponse(BaseModel):
    """Prediction output schema."""
    churn_probability: float = Field(..., description="Probability of customer churning (0-1)")
    no_churn_probability: float = Field(..., description="Probability of customer not churning (0-1)")
    prediction: str = Field(..., description="Binary prediction: 'Churn' or 'No Churn'")
    risk_level: str = Field(..., description="Risk level: 'Low', 'Medium', or 'High'")


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict customer churn probability.

    Args:
        customer: Customer data in JSON format

    Returns:
        Prediction probabilities and risk level
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.dict()])

        # Preprocess data
        X, _ = preprocessor.prepare_data(input_data, fit=False)

        # Make prediction
        prediction_proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]

        # Calculate probabilities
        no_churn_prob = float(prediction_proba[0])
        churn_prob = float(prediction_proba[1])

        # Determine risk level
        if churn_prob < 0.3:
            risk_level = "Low"
        elif churn_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Return response
        return PredictionResponse(
            churn_probability=churn_prob,
            no_churn_probability=no_churn_prob,
            prediction="Churn" if prediction == 1 else "No Churn",
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for multiple customers.

    Args:
        customers: List of customer data

    Returns:
        List of predictions
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        for customer in customers:
            # Convert to DataFrame
            input_data = pd.DataFrame([customer.dict()])

            # Preprocess and predict
            X, _ = preprocessor.prepare_data(input_data, fit=False)
            prediction_proba = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]

            churn_prob = float(prediction_proba[1])

            # Determine risk level
            if churn_prob < 0.3:
                risk_level = "Low"
            elif churn_prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            predictions.append({
                "churn_probability": churn_prob,
                "no_churn_probability": float(prediction_proba[0]),
                "prediction": "Churn" if prediction == 1 else "No Churn",
                "risk_level": risk_level
            })

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
