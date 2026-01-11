#  Heart Disease Prediction – End-to-End Machine Learning Project

##  Overview
This project implements an **end-to-end machine learning pipeline** to predict the likelihood of heart disease using clinical and lifestyle features.  
It covers the complete ML lifecycle — from **EDA and statistical testing** to **model deployment using Streamlit and Docker**.

---

##  Problem Statement
Early detection of heart disease is critical for preventive healthcare.  
The goal of this project is to build a reliable and interpretable model that predicts heart disease risk based on patient health indicators.

---

##  Dataset
- **1000 patient records**
- Combination of **clinical** and **lifestyle** features
- Target variable: `Heart Disease (0 = No, 1 = Yes)`

---

##  Exploratory Data Analysis (EDA)
Key findings:
- **Age** and **cholesterol** are the strongest predictors
- **Smoking** is the only statistically significant categorical variable
- **Gender** and **family history** show weak standalone effects
- Most features contribute meaningfully when combined

---

##  Statistical Testing
- **Chi-square tests** for categorical features
- **Independent t-tests** for numerical features
- Findings validated using **logistic regression coefficients**

---

##  Feature Engineering
- Missing value handling (median / most frequent)
- One-hot encoding for categorical variables
- Feature scaling using standardization
- Pipeline-based preprocessing for reproducibility

---

## 🤖 Modeling
Models evaluated:
- Logistic Regression 
- Decision Tree 
- Random Forest
- Support Vector Machine (SVM)


---

##  Model Selection
**Logistic Regression** was selected because it:
- Generalizes well
- Avoids overfitting
- Is highly interpretable
- Aligns with statistical and medical insights

---

##  Deployment
- Interactive **Streamlit web application**
- Modular **predict pipeline**
- Uses saved **preprocessor** and **trained model**
- Ready for real-time inference

---

##  Docker Deployment

This project is fully containerized using Docker to ensure **reproducibility and easy deployment** across environments.

> **Note:** Docker is used **only for inference and deployment**.  
> Model training is performed separately before containerization.

###  Prerequisites
- Docker Desktop installed and running  
   https://www.docker.com/products/docker-desktop/

---

###  Build the Docker Image
From the project root directory:

```bash
docker build -t heart-disease-app .

# ❤️ Heart Disease Prediction – End-to-End Machine Learning Project

## 📌 Overview
This project implements an **end-to-end machine learning pipeline** to predict the likelihood of heart disease using clinical and lifestyle features.  
It covers the complete ML lifecycle — from **EDA and statistical testing** to **model deployment using Streamlit and Docker**.

---

## 🎯 Problem Statement
Early detection of heart disease is critical for preventive healthcare.  
The goal of this project is to build a reliable and interpretable model that predicts heart disease risk based on patient health indicators.

---

## 📊 Dataset
- **1000 patient records**
- Combination of **clinical** and **lifestyle** features
- Target variable: `Heart Disease (0 = No, 1 = Yes)`

---

## 🔍 Exploratory Data Analysis (EDA)
Key findings:
- **Age** and **cholesterol** are the strongest predictors
- **Smoking** is the only statistically significant categorical variable
- **Gender** and **family history** show weak standalone effects
- Most features contribute meaningfully when combined

---

## 📐 Statistical Testing
- **Chi-square tests** for categorical features
- **Independent t-tests** for numerical features
- Findings validated using **logistic regression coefficients**

---

## 🛠️ Feature Engineering
- Missing value handling (median / most frequent)
- One-hot encoding for categorical variables
- Feature scaling using standardization
- Pipeline-based preprocessing for reproducibility

---

## 🤖 Modeling
Models evaluated:
- Logistic Regression ✅ **(Selected)**
- Decision Tree ❌ (Rejected – overfitting)
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 🏆 Model Selection
**Logistic Regression** was selected because it:
- Generalizes well
- Avoids overfitting
- Is highly interpretable
- Aligns with statistical and medical insights

---

## 🚀 Deployment
- Interactive **Streamlit web application**
- Modular **predict pipeline**
- Uses saved **preprocessor** and **trained model**
- Ready for real-time inference

---

## 🐳 Docker Deployment

This project is fully containerized using Docker to ensure **reproducibility and easy deployment** across environments.

> **Note:** Docker is used **only for inference and deployment**.  
> Model training is performed separately before containerization.

###  Prerequisites
- Docker Desktop installed and running  
  👉 https://www.docker.com/products/docker-desktop/

---

###  Build the Docker Image
From the project root directory:

```bash
docker build -t heart-disease-app .
---

##  Run Locally (Without Docker)

Follow these steps to run the Heart Disease Prediction application locally on your machine.

---

###  Prerequisites
- Python **3.9 or above**
- Git installed

---

###  Clone the Repository
```bash
git clone https://github.com/<your-username>/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
