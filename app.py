import streamlit as st
import joblib
import pandas as pd

model = joblib.load('KNN_heart_model.pkl')
scaler = joblib.load('sc.pkl')
expected_columns = joblib.load('columns.pkl')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️‍🩹", layout='wide')
st.title("Heart Disease Predictor🫀")

st.markdown("Provide the following details to predict the likelihood of heart disease:")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F']) 
chest_pain_type = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_heart_rate = st.number_input("Max Heart Rate Achieved", 60, 220, 150)   
exercise_angina = st.selectbox("Exercise Angina", ['Yes', 'No'])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the ST segment", ['Up', 'Flat', 'Down'])

if st.button("Predict"): 
    input_data = {
        'Age': age,
        'Sex_'+ sex: 1, 
        'ChestPainType_'+ chest_pain_type: 1,
        'RestingBP': resting_blood_pressure, 
        'Cholesterol': cholesterol,
        'FastingBS_'+ fasting_bs: 1,
        'RestingECG_'+ resting_ecg: 1,
        'MaxHR': max_heart_rate,
        'ExerciseAngina_'+ exercise_angina: 1,
        'Oldpeak': oldpeak,
        'ST_Slope_'+ st_slope: 1
    }
    input_df = pd.DataFrame([input_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.error("⚠️The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("💚The model predicts that you are unlikely to have heart disease. However, please maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.")