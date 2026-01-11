import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title(" Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")


age = st.number_input("Age", min_value=1, max_value=120, value=50)

gender = st.selectbox("Gender", ["Male", "Female"])

cholesterol = st.number_input("Cholesterol", min_value=50, max_value=500, value=200)

blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)

heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=70)

smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "High", "Unknown"])

exercise_hours = st.number_input("Exercise Hours (per week)", min_value=0, max_value=20, value=3)

family_history = st.selectbox("Family History of Heart Disease", ["Yes", "No"])

diabetes = st.selectbox("Diabetes", ["Yes", "No"])

obesity = st.selectbox("Obesity", ["Yes", "No"])

stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5)

blood_sugar = st.number_input("Blood Sugar", min_value=50, max_value=300, value=120)

angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Heart Disease"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Cholesterol": cholesterol,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Smoking": smoking,
        "Alcohol Intake": alcohol,
        "Exercise Hours": exercise_hours,
        "Family History": family_history,
        "Diabetes": diabetes,
        "Obesity": obesity,
        "Stress Level": stress_level,
        "Blood Sugar": blood_sugar,
        "Exercise Induced Angina": angina,
        "Chest Pain Type": chest_pain
    }])

    pipeline = PredictPipeline()
    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error(" High Risk of Heart Disease")
    else:
        st.success(" Low Risk of Heart Disease")
