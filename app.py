import streamlit as st
import pandas as pd
import numpy as np
import joblib

heart_failure_model = joblib.load('heart_failure_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Failure Prediction App")
st.background_color = "#36953f"
age =  st.slider("Age", 20, 100, 50)
anemia = st.selectbox("Anemia(0=No, 1=Yes )", [0, 1])
creatinine = st.number_input("Creatinine Phosphokinase (CPK) level", min_value=0, max_value=10000, value=100)   
diabetes = st.radio("Diabetes(0=No, 1=Yes )", [0, 1])
ef = st.number_input("Ejection Fraction (EF) percentage", min_value=0, max_value=100, value=50)
hbp = st.radio("High Blood Pressure(0=No, 1=Yes )", [0, 1])
platelets = st.number_input("Platelets count (in 10^3/μL)", min_value=0, max_value=1000, value=250)
serum_creatinine = st.number_input("Serum Creatinine level (mg/dL  )", min_value=0.0, max_value=10.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium level (mEq/L)", min_value=100, max_value=200, value=135)
sex = st.radio("Sex(0=Female, 1=Male )", [0, 1])
smoking = st.radio("Smoking(0=No, 1=Yes )", [0, 1])
time = st.number_input("Follow-up period (in days)", min_value=0, max_value=1000, value=100)    

if st.button("Predict"):
    input_data = np.array([[age, anemia, creatinine, diabetes, ef, hbp, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])
    input_data_scaled = scaler.transform(input_data)
    prediction = heart_failure_model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.error("The patient is at risk of heart failure.")
    else:
        st.success("The patient is not at risk of heart failure.")
