# Customer Churn Prediction for Telco - End to End workflow

This project demonstrates a complete machine learning workflow for predicting customer churn of a telecommunications company, from data acquisition to model deployment and monitoring.

## Project Structure

```
churn_prediction/
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned, transformed data
├── notebooks/        # Jupyter notebooks for analysis
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_improved_xgboost_tuning.ipynb
├── src/              # Python modules
│   ├── 04_model_deployment.py
│   ├── client_example.py
├── models/           # Saved model files
├── logs/             # Prediction logs for monitoring
├── README.md         # Project documentation
```

### Project Steps
1. Data Acquisition and Exploration

- Loaded the Telco Customer Churn dataset from Kaggle
- Explored data structure and distributions
- Identified key features and relationships
- Detected initial patterns between features and churn

2. Data Preprocessing and Feature Engineering

- Handled missing values in TotalCharges
- Encoded categorical variables
- Created new features:

   - Tenure groups
   - Total services count
   - Charge evolution metrics


- Prepared data for modeling

3. Model Training and Evaluation

- Trained multiple models:

   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost


- Evaluated models using metrics:

   - Accuracy, Precision, Recall, F1 Score, ROC AUC


- Selected best model based on F1 score
- Analyzed feature importance

4. Model Improvement

- Addressed class imbalance using scale_pos_weight
- Performed extensive hyperparameter tuning
- Improved F1 score significantly

5. Model Deployment

- Created a Flask API for serving predictions
- Implemented endpoint for single and batch predictions
- Added health check and monitoring capabilities
- Developed example client application

6. Retention Strategy Implementation

- Translated model predictions into actionable insights
- Developed tiered intervention strategies based on churn risk
- Created cost-benefit analysis framework
- Implemented monitoring and evaluation system

## How to Use This Project

### Setup

1. Clone the repository
2. Create a virtual environment:

```
    python -m venv churn_env
    source churn_env/bin/activate
```
Install dependencies:
```
    pip install -r requirements.txt (requirements.txt file is not available in this repository)
```
## Running the Notebooks

Execute the notebooks in sequence to reproduce the analysis:

1. Data Acquisition
2. Data Preprocessing
3. Model Training

## Model Deployment

1. Train and save the model using the notebooks
2. Start the Flask API:
```
    python src/04_model_deployment.py
```
## Using API Client

Use the client example to test the API:
```
    python src/client_example.py
```
Othewise make HTTP requests directly:
```
    curl -X POST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d '{"customerID":"Sample-1234", ...}
```
## Business Value

This churn prediction system provides:

1. Early Warning System: Identify at-risk customers before they leave
2. Targeted Interventions: Focus retention efforts on highest-risk customers
3. Cost Efficiency: Optimize retention budget with risk-based prioritization
4. Customer Insights: Understand key factors driving churn
5. ROI Measurement: Track the effectiveness of retention campaigns


## Future Steps / Improvements

- Integrate with CRM systems / databases for automated interventions
- Implement A/B testing before fully deploying the model to measure its impact
- Create a real-time monitoring dashboard
- Expand model to predict lifetime value alongside churn risk
- Create an automated retraining pipeline for model updates (avoid model drift)
