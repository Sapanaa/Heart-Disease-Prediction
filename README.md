# Heart-Disease-Prediction



# Heart Disease Prediction – End-to-End ML Project

## Problem Statement
Early detection of heart disease using clinical and lifestyle features.

## Dataset
- 1000 patient records
- Clinical + lifestyle indicators

## Exploratory Data Analysis
- Age and cholesterol are strongest predictors
- Smoking is the only statistically significant categorical variable
- Gender and family history show weak standalone effects

## Statistical Testing
- Chi-square tests for categorical features
- Independent t-tests for numeric features
- Results validated via logistic regression coefficients

## Feature Engineering
- Missing value handling
- One-hot encoding
- Standardization

## Modeling
Models tested:
- Logistic Regression (Selected)
- Decision Tree (Rejected – Overfitting)
- Random Forest
- SVM
- KNN

## Model Selection
Logistic Regression chosen due to:
- Strong generalization
- Interpretability
- Consistency with statistical analysis

## Deployment
- Streamlit web application
- Predict pipeline using saved model and preprocessor

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py















python -m venv venv


venv\Scripts\activate


