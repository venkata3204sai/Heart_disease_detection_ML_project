import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('models.pkl', 'rb') as file:
    model = pickle.load(file)

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.title('Heart Disease Prediction App')

age = st.number_input("Age", min_value=18, max_value=100, value=30)

sex = st.selectbox("Sex", options=["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type", 
    options=["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)", 
    min_value=50, max_value=250, value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)", 
    min_value=100, max_value=600, value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", 
    options=[0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG Result", 
    options=["Normal", "ST", "LVH"]
)

max_hr = st.number_input(
    "Maximum Heart Rate Achieved", 
    min_value=60, max_value=220, value=140
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina", 
    options=["Y", "N"]
)

oldpeak = st.number_input(
    "Oldpeak (ST depression induced by exercise)", 
    min_value=0.0, max_value=10.0, step=0.1, value=1.0
)

st_slope = st.selectbox(
    "Slope of the Peak Exercise ST Segment", 
    options=["Up", "Flat", "Down"]
)

if st.button('Predict'):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })
    
    input_processed = pipeline.transform(input_data)
    
    prediction = model.predict(input_processed)

    if prediction[0] == 1:
        st.error('High risk of Heart Disease')
    else:
        st.success('Low risk of Heart Disease')
