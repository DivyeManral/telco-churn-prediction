import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load Model & Encoders
# -------------------------------
preprocessor = pickle.load(open("app/encoders.pkl", "rb"))
# Load model
model = pickle.load(open("customer_churn_model.pkl", "rb"))
# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.markdown("""
# 📊 Telco Customer Churn Prediction
### Predict whether a customer is likely to churn
""")

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Customer Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    st.subheader("📊 Account Info")
    tenure = st.slider("Tenure (months)", 1, 72, 12)

    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Churn"):

    # Create DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # -------------------------------
    # Encoding using your encoders dict
    # -------------------------------
    input_data['gender'] = preprocessor['gender'].transform(input_data['gender'])
    input_data['Partner'] = preprocessor['Partner'].transform(input_data['Partner'])
    input_data['Dependents'] = preprocessor['Dependents'].transform(input_data['Dependents'])
    input_data['PhoneService'] = preprocessor['PhoneService'].transform(input_data['PhoneService'])
    input_data['MultipleLines'] = preprocessor['MultipleLines'].transform(input_data['MultipleLines'])
    input_data['InternetService'] = preprocessor['InternetService'].transform(input_data['InternetService'])
    input_data['OnlineSecurity'] = preprocessor['OnlineSecurity'].transform(input_data['OnlineSecurity'])
    input_data['OnlineBackup'] = preprocessor['OnlineBackup'].transform(input_data['OnlineBackup'])
    input_data['DeviceProtection'] = preprocessor['DeviceProtection'].transform(input_data['DeviceProtection'])
    input_data['TechSupport'] = preprocessor['TechSupport'].transform(input_data['TechSupport'])
    input_data['StreamingTV'] = preprocessor['StreamingTV'].transform(input_data['StreamingTV'])
    input_data['StreamingMovies'] = preprocessor['StreamingMovies'].transform(input_data['StreamingMovies'])
    input_data['Contract'] = preprocessor['Contract'].transform(input_data['Contract'])
    input_data['PaperlessBilling'] = preprocessor['PaperlessBilling'].transform(input_data['PaperlessBilling'])
    input_data['PaymentMethod'] = preprocessor['PaymentMethod'].transform(input_data['PaymentMethod'])

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model["model"].predict(input_data)

    # -------------------------------
    # Output
    # -------------------------------
    st.markdown("---")

    if prediction[0] == 1:
        st.markdown("## ⚠️ High Risk of Churn")
        st.error("Customer is likely to churn. Consider retention strategies.")
    else:
        st.markdown("## ✅ Low Risk of Churn")
        st.success("Customer is likely to stay.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👨‍💻 Built by Divye Manral | Machine Learning Project")
