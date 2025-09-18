import streamlit as st
import pandas as pd
import joblib

# Path to the tuned model
MODEL_PATH = r"C:\Users\moham\miniconda3\envs\heart_ml\models\final_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("❤️ Heart Disease Risk Prediction App")
st.write("Fill in patient details below to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=54)
sex = st.selectbox("Sex", (0, 1), format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type (0–3)", (0,1,2,3))
trestbps = st.number_input("Resting blood pressure (mmHg)", min_value=50, max_value=250, value=130)
chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", (0,1), format_func=lambda x: "No" if x==0 else "Yes")
restecg = st.selectbox("Resting ECG (0,1,2)", (0,1,2))
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise induced angina", (0,1), format_func=lambda x: "No" if x==0 else "Yes")
oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of the ST segment (0–2)", (0,1,2))
ca = st.selectbox("Number of major vessels (0–3)", (0,1,2,3))
thal = st.selectbox("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", (1,2,3))

# Prepare input as dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Model predicts **Heart Disease** (probability: {prob:.2f})")
    else:
        st.success(f"✅ Model predicts **No Heart Disease** (probability: {prob:.2f})")
