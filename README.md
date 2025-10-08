# Customer Churn Prediction

ML system for predicting customer churn in telecom using XGBoost + FastAPI.

## Performance

- **Accuracy**: 77.93%
- **ROC-AUC**: 0.82
- **F1 Score**: 0.61
- **Training Time**: ~3-4 minutes (4 baseline models + GridSearchCV with 324 configs)

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Train the Model

```bash
python src/train_model.py
```

This will:
- Load and preprocess the data
- Train baseline models for comparison (LR, RF, GB, XGBoost)
- Run hyperparameter tuning on XGBoost
- Save the best model to `models/`

Takes about 3-4 minutes on my machine (8-core CPU).

**Note**: You can set `COMPARE_BASELINES = False` in the script to skip baseline comparison and reduce training time to ~2-3 minutes.

### Start the API

```bash
python src/app.py
```

API will be running at http://localhost:8000

You can test it with the interactive docs at http://localhost:8000/docs

### Make a Prediction

**Python example:**
```python
import requests

# Example high-risk customer
customer = {
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

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
```

**Response:**
```json
{
  "churn_probability": 0.926,
  "no_churn_probability": 0.074,
  "prediction": "Churn",
  "risk_level": "High"
}
```

You can also use cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer_data.json
```

## Project Structure

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.joblib          # Trained model
│   ├── preprocessor.joblib         # Preprocessing pipeline
│   ├── metrics.json
│   └── feature_importance.json
├── notebooks/
│   ├── 01_exploratory_data_analysis_executed.ipynb
│   ├── 02_model_training_evaluation_executed.ipynb
│   └── figures/                     # 10 visualizations
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── app.py
├── test_api.py
└── requirements.txt
```

## Model Details

**Dataset**: Telco Customer Churn (7,043 customers, 19 features)

**Preprocessing**:
- Created 5 new features: `tenure_group`, `avg_monthly_spend`, `total_services`, `has_internet`, `payment_electronic_check`
- Used SMOTE to handle class imbalance (27% → 50% balanced training)
- Label encoding for categorical variables + StandardScaler for numerical

**Model Selection**:
I tested 4 algorithms and XGBoost performed best:
- Logistic Regression: 74.9% accuracy
- Random Forest: 76.4%
- Gradient Boosting: 75.7%
- **XGBoost: 77.6%** ← selected this

Then ran GridSearchCV (324 configurations) which improved it to 77.93%.

**Top 5 Features**:
1. Contract type (41.6%) - month-to-month contracts have way higher churn
2. Tenure group (6.8%) - new customers churn much more
3. Online Security (6.3%)
4. Tech Support (5.1%)
5. Internet Service type (4.3%)

## API Endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/` | GET | API info |
| `/health` | GET | Check if model is loaded |
| `/predict` | POST | Predict single customer |
| `/predict/batch` | POST | Predict multiple customers |
| `/docs` | GET | Interactive API documentation |

## Notebooks

I've included 2 notebooks with all my analysis:

**01_exploratory_data_analysis_executed.ipynb**
- Data exploration, churn distribution
- Feature correlations and relationships

**02_model_training_evaluation_executed.ipynb**
- Model comparison charts
- ROC curves, learning curves
- Confusion matrix and feature importance plots

## Testing

```bash
python test_api.py
```

Tests the health check, single prediction, and batch prediction endpoints.

## Technical Details

See `REPORT.md` for detailed explanations of:
- Why I chose this dataset
- Feature engineering decisions
- Model architecture and tuning approach
- Challenges I ran into
- Business insights from the model

## License

MIT
