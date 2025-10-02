"""
Test script for the Customer Churn Prediction API
"""
import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test data for prediction
    customer_data = {
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
    
    try:
        print("Testing Customer Churn Prediction API...")
        print("=" * 50)
        
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Health check failed")
            return
        
        print("\n2. Testing single prediction...")
        response = requests.post(f"{base_url}/predict", json=customer_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful")
            print(f"   Customer ID: {result['customer_id']}")
            print(f"   Churn Probability: {result['churn_probability']:.4f}")
            print(f"   Churn Prediction: {result['churn_prediction']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print("❌ Single prediction failed")
            print(f"   Error: {response.text}")
        
        print("\n3. Testing feature importance...")
        response = requests.get(f"{base_url}/feature-importance")
        if response.status_code == 200:
            result = response.json()
            print("✅ Feature importance retrieved")
            print(f"   Model: {result['model_name']}")
            print("   Top 5 features:")
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:5]):
                print(f"     {i+1}. {feature}: {importance:.4f}")
        else:
            print("❌ Feature importance failed")
            print(f"   Error: {response.text}")
        
        print("\n4. Testing batch prediction...")
        batch_data = {
            "customers": [customer_data, {**customer_data, "Contract": "Month-to-month", "tenure": 2}]
        }
        response = requests.post(f"{base_url}/batch-predict", json=batch_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful")
            print(f"   Total customers: {result['summary']['total_customers']}")
            print(f"   Predicted churn count: {result['summary']['predicted_churn_count']}")
            print(f"   Predicted churn rate: {result['summary']['predicted_churn_rate']:.4f}")
        else:
            print("❌ Batch prediction failed")
            print(f"   Error: {response.text}")
        
        print("\n🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the API server is running")
        print("   Start the server with: python api/main.py")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_api()
