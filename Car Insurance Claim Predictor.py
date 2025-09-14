import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import joblib


# اختيارات المستخدم
age = st.selectbox("Age group", ["16-25", "26-39", "40-64", "65+"])
gender = st.selectbox("Gender", ["Female", "Male"])
driving_exp = st.selectbox("Driving Experience", ["0-9", "10-19", "20-29", "30+"])
education = st.selectbox("Education", ["No education", "High school", "University"])
income = st.selectbox("Income", ["Poverty", "Working class", "Middle class", "Upper class"])
credit_score = st.slider("Credit Score", 0.0, 1.0, 0.5)
vehicle_ownership = st.radio("Vehicle Ownership", ["Does not own", "Owns"])
vehicle_year = st.radio("Vehicle Year", ["Before 2015", "2015 or later"])
married = st.radio("Married?", ["No", "Yes"])
children = st.number_input("Number of Children", 0, 10, 0)
annual_mileage = st.number_input("Annual Mileage", 0, 50000, 10000)
vehicle_type = st.radio("Vehicle Type", ["Sedan", "Sports car"])
speeding = st.number_input("Speeding Violations", 0, 20, 0)
duis = st.number_input("DUIs", 0, 10, 0)
accidents = st.number_input("Past Accidents", 0, 10, 0)

# تحويل الاختيارات لأرقام زي الداتا
mapping = {
    "16-25":0, "26-39":1, "40-64":2, "65+":3,
    "Female":0, "Male":1,
    "0-9":0, "10-19":1, "20-29":2, "30+":3,
    "No education":0, "High school":1, "University":2,
    "Poverty":0, "Working class":1, "Middle class":2, "Upper class":3,
    "Does not own":0, "Owns":1,
    "Before 2015":0, "2015 or later":1,
    "No":0, "Yes":1,
    "Sedan":0, "Sports car":1
}

input_data = np.array([[
    mapping[age], mapping[gender], mapping[driving_exp],
    mapping[education], mapping[income], credit_score,
    mapping[vehicle_ownership], mapping[vehicle_year],
    mapping[married], children, 0,  # postal_code = 0 (مش مستخدم هنا)
    annual_mileage, mapping[vehicle_type],
    speeding, duis, accidents
]])

# تحميل الموديل
model = joblib.load("insurance_model.pkl")

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("This client is likely to make a claim!")
    else:
        st.success("This client is unlikely to make a claim.")

