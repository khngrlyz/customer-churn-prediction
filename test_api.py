"""
Test script for the Customer Churn Prediction API.
Demonstrates how to interact with the deployed model.
"""

import requests
import json


# API endpoint
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test single customer prediction."""
    print("Testing single prediction...")

    # Example customer data (high churn risk)
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_low_churn_customer():
    """Test customer with low churn risk."""
    print("Testing low churn risk customer...")

    # Example customer data (low churn risk)
    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 105.00,
        "TotalCharges": 6300.00
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_prediction():
    """Test batch prediction with multiple customers."""
    print("Testing batch prediction...")

    customers = [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.00,
            "TotalCharges": 70.00
        },
        {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 36,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 55.00,
            "TotalCharges": 1980.00
        }
    ]

    response = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("="*60)
    print("CUSTOMER CHURN PREDICTION API - TEST SCRIPT")
    print("="*60)
    print()

    try:
        # Run all tests
        test_health_check()
        test_single_prediction()
        test_low_churn_customer()
        test_batch_prediction()

        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        print("Make sure the server is running: python src/app.py")
    except Exception as e:
        print(f"ERROR: {e}")
