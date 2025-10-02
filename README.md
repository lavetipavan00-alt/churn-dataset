# Customer Churn Prediction System

A complete end-to-end machine learning system for predicting customer churn in telecommunications services using the IBM Telco Customer Churn dataset.

## Project Overview

This project implements a comprehensive customer churn prediction system including:
- **Data Analysis & Preprocessing**: EDA, feature engineering, data cleaning
- **Model Development**: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning
- **Model Interpretation**: Feature importance analysis and SHAP explanations
- **API Development**: FastAPI REST API for predictions
- **Business Insights**: Actionable recommendations for customer retention

## Project Structure

```
Ace/
├── data/
│   ├── download_dataset.py      # Script to download the dataset
│   └── Telco-Customer-Churn.csv # Main dataset
├── src/
│   └── churn_model_pipeline.py  # Complete ML pipeline
├── api/
│   └── main.py                  # FastAPI application
├── models/
│   ├── best_model.joblib        # Trained best model
│   ├── preprocessor.joblib      # Data preprocessor
│   ├── feature_names.json       # Feature names
│   └── feature_importance.json  # Feature importance scores
├── plots/
│   ├── feature_importance.png   # Feature importance visualization
│   └── model_comparison.png     # Model performance comparison
├── reports/
│   └── business_insights.json   # Business recommendations
├── customer_churn_analysis.ipynb # Jupyter notebook analysis
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- pip (Python package installer)

## Installation & Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
```bash
python data/download_dataset.py
```

### Step 3: Run Complete ML Pipeline
```bash
python src/churn_model_pipeline.py
```

### Step 4: Start the API Server
```bash
cd api
python main.py
```
Or using uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Access the API
- API Documentation: http://localhost:8000/docs
- API Root: http://localhost:8000/

## API Endpoints

### 1. Single Customer Prediction
**POST** `/predict`

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Male",
       "SeniorCitizen": "No",
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 24,
       "PhoneService": "Yes",
       "MultipleLines": "Yes",
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
       "MonthlyCharges": 79.85,
       "TotalCharges": 1889.5
     }'
```

### 2. Batch Predictions
**POST** `/batch-predict`

### 3. Feature Importance
**GET** `/feature-importance`

Example:
```bash
curl -X GET "http://localhost:8000/feature-importance"
```

### 4. Health Check
**GET** `/health`

## Model Performance

The pipeline trains and compares three models:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|---------|----------|
| Logistic Regression | 0.8465 | - | - | - | - |
| Random Forest | 0.8428 | - | - | - | - |
| XGBoost | 0.8459 | - | - | - | - |

**Best Model**: Logistic Regression (ROC-AUC: 0.8465)

## Key Findings & Business Insights

### High-Risk Churn Factors:
1. **Contract Type**: Month-to-month contracts have highest churn rates
2. **Tenure**: New customers (0-12 months) are most likely to churn
3. **Payment Method**: Electronic check payments increase churn risk
4. **Internet Service**: Fiber optic customers have higher churn rates
5. **Family Status**: Customers without partners/dependents are higher risk

### Business Recommendations:
1. **Focus on Month-to-Month Customers**: Implement retention programs
2. **New Customer Onboarding**: Intensive support in first 12 months
3. **Payment Method Incentives**: Encourage automatic payments
4. **Fiber Optic Service**: Improve quality and customer support
5. **Family Packages**: Develop family-oriented service offerings

## Running Jupyter Notebook Analysis

```bash
jupyter notebook customer_churn_analysis.ipynb
```

## Testing the API

### Using Python requests:
```python
import requests

# Single prediction
data = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.20,
    "TotalCharges": 542.4
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## File Descriptions

- **`src/churn_model_pipeline.py`**: Complete ML pipeline with data preprocessing, model training, evaluation, and saving
- **`api/main.py`**: FastAPI application with prediction endpoints
- **`data/download_dataset.py`**: Downloads the Telco Customer Churn dataset
- **`customer_churn_analysis.ipynb`**: Detailed Jupyter notebook analysis
- **`requirements.txt`**: All Python dependencies

## Technical Features

- **Data Preprocessing**: Handles missing values, feature engineering
- **Class Imbalance**: SMOTE implementation for balanced training
- **Hyperparameter Tuning**: Grid/Random search for optimal parameters
- **Model Validation**: Cross-validation and proper train/test splits
- **Feature Engineering**: Creates meaningful derived features
- **Model Interpretation**: Feature importance and SHAP analysis
- **API Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error handling and logging

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Not Found**: Run the download script
   ```bash
   python data/download_dataset.py
   ```

3. **API Not Starting**: Check if port 8000 is available
   ```bash
   uvicorn api.main:app --port 8001
   ```

4. **Memory Issues**: Reduce sample sizes in SHAP analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes and uses the IBM Telco Customer Churn dataset.

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.
