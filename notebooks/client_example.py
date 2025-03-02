"""
Customer Churn Prediction API Client Example
-------------------------------------------
This script demonstrates how to:
1. Connect to the churn prediction API
2. Send customer data for prediction
3. Handle the response
4. Implement a proactive retention strategy based on predictions
"""

import requests
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# API endpoint
API_URL = "http://localhost:5000"

def check_api_health():
    """Check if the API is up and running"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"API Status: {health_data['status']}")
            print(f"Model Loaded: {health_data['model_loaded']}")
            return True
        else:
            print(f"API Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False


def predict_customer_churn(customer_data):
    """
    Send customer data to the API and get churn prediction
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error making prediction request: {e}")
        return None
    

def batch_predict_churn(customers_data):
    """
    Send multiple customer records for batch prediction
    """
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=customers_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error making batch prediction request: {e}")
        return None
    

def create_retention_action(prediction):
    """
    Create a tailored retention strategy based on the prediction
    """
    customer_id = prediction['customer_id']
    churn_prob = prediction['churn_probability']
    is_high_risk = prediction['is_high_risk']
    
    # Define actions based on risk level
    if is_high_risk:
        action = {
            'customer_id': customer_id,
            'risk_level': 'High',
            'churn_probability': churn_prob,
            'actions': [
                'Direct call from retention specialist',
                'Personalized discount offer (15-20 of MRC)',
                'Service upgrade at current price, especially if fiber customer',
                'Address pain points identified in last survey'
            ],
            'urgency': 'Immediate (within 24 hours)',
            'expected_cost': '$50-100 in discounts/upgrades',
            'expected_benefit': 'Potential $800+ lifetime value saved'
        }
    elif churn_prob > 0.4:
        action = {
            'customer_id': customer_id,
            'risk_level': 'Medium',
            'churn_probability': churn_prob,
            'actions': [
                'Email with targeted offer',
                'Customer satisfaction survey',
                'Modest loyalty discount (5-10%)'
            ],
            'urgency': 'This week',
            'expected_cost': '$20-50 in discounts',
            'expected_benefit': 'Potential $800+ lifetime value saved'
        }
    else:
        action = {
            'customer_id': customer_id,
            'risk_level': 'Low',
            'churn_probability': churn_prob,
            'actions': [
                'Include in regular loyalty program',
                'Send satisfaction survey in next batch'
            ],
            'urgency': 'Routine',
            'expected_cost': 'Minimal',
            'expected_benefit': 'Customer retention and satisfaction'
        }
    
    return action


def analyze_batch_results(predictions):
    """Analyze a batch of prediction results"""
    if not predictions:
        return
    
    df = pd.DataFrame(predictions)
    
    print("\nPrediction Summary:")
    print(f"Total customers analyzed: {len(df)}")
    print(f"Predicted to churn: {df['churn_prediction'].sum()} ({df['churn_prediction'].mean():.1%})")
    print(f"High-risk customers: {df['is_high_risk'].sum()} ({df['is_high_risk'].mean():.1%})")
    
    # Risk distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['churn_probability'], bins=20, kde=True)
    plt.title('Distribution of Churn Probability')
    plt.xlabel('Churn Probability')
    plt.ylabel('Count')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Churn Threshold')
    plt.axvline(x=0.7, color='darkred', linestyle='--', label='High Risk Threshold')
    plt.legend()
    plt.show()
    
    # Calculate potential financial impact
    avg_customer_value = 800  # Example: average customer lifetime value
    potential_loss = df['churn_probability'].sum() * avg_customer_value
    print(f"\nPotential revenue at risk: ${potential_loss:.2f}")
    
    # If we intervene with high-risk customers and save 30% of them
    high_risk_customers = df[df['is_high_risk']]
    intervention_success_rate = 0.3
    saved_revenue = len(high_risk_customers) * intervention_success_rate * avg_customer_value
    
    print(f"If we focus on high-risk customers with a {intervention_success_rate:.0%} success rate:")
    print(f"Estimated saved revenue: ${saved_revenue:.2f}")


def demo_client():
    # Check if API is available
    if not check_api_health():
        print("API is not available. Please start the API server first.")
        return
    
    print("\n=== Single Customer Prediction ===")
    
    # Example customer data
    customer = {
        'customerID': 'Samp-1234',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 36,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 105.70,
        'TotalCharges': 3102.35
    }
    
    # Make prediction
    print(f"Predicting churn for customer {customer['customerID']}...")
    prediction = predict_customer_churn(customer)
    
    if prediction:
        print("\nPrediction Result:")
        print(f"Churn Probability: {prediction['churn_probability']:.2%}")
        print(f"Churn Prediction: {'Yes' if prediction['churn_prediction'] == 1 else 'No'}")
        print(f"High Risk: {'Yes' if prediction['is_high_risk'] else 'No'}")
        
        # Get retention action
        action = create_retention_action(prediction)
        
        print("\nRecommended Retention Strategy:")
        print(f"Risk Level: {action['risk_level']}")
        print(f"Actions:")
        for item in action['actions']:
            print(f"  - {item}")
        print(f"Urgency: {action['urgency']}")
        print(f"Expected Cost: {action['expected_cost']}")
        print(f"Expected Benefit: {action['expected_benefit']}")
    
    print("\n=== Batch Prediction Demo ===")
    
    # Create a batch of sample customers
    batch_customers = []
    
    # High-risk example
    high_risk = {
        'customerID': 'NG-3214',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.20,
        'TotalCharges': 455.60
    }
    batch_customers.append(high_risk)
    
    # Medium-risk example
    medium_risk = {
        'customerID': 'MM-0525',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 16,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 54.80,
        'TotalCharges': 243.00
    }
    batch_customers.append(medium_risk)
    
    # Low-risk example
    low_risk = {
        'customerID': 'SH-1298',
        'gender': 'Male',
        'SeniorCitizen': 1,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 120.15,
        'TotalCharges': 5407.12
    }
    batch_customers.append(low_risk)
    
    # Make batch prediction
    print(f"Making batch predictions for {len(batch_customers)} customers...")
    batch_predictions = batch_predict_churn(batch_customers)
    
    if batch_predictions:
        # Analyze results
        analyze_batch_results(batch_predictions)
        
        # Print individual results
        print("\nIndividual Retention Strategies:")
        for prediction in batch_predictions:
            action = create_retention_action(prediction)
            print(f"\nCustomer {prediction['customer_id']}:")
            print(f"  Risk Level: {action['risk_level']} (Churn Prob: {prediction['churn_probability']:.2%})")
            print(f"  Primary Action: {action['actions'][0]}")
            print(f"  Urgency: {action['urgency']}")


if __name__ == "__main__":
    print("Churn Prediction Client Demo")
    print("----------------------------")
    demo_client()