import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. Page Configuration & Assets Loading
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Customer Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_assets():
    rf_model = joblib.load("best_rf_pipeline.pkl")
    ada_model = joblib.load("best_ada_pipeline.pkl")
    features = joblib.load("ecommerce_customer_churn_features.pkl")
    return rf_model, ada_model, features

rf_model, ada_model, features = load_assets()

# ---------------------------------------------------------
# 2. Main UI & Sidebar Settings
# ---------------------------------------------------------
st.title("🛒 E-commerce Customer Churn Prediction")
st.markdown("Enter customer details below and select a Machine Learning model to predict churn probability.")

st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["Random Forest Classifier", "AdaBoost Classifier"]
)

st.write("---")
st.subheader("📋 Customer Details Input")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0.0, max_value=100.0, value=9.0, step=1.0)
    warehouse_to_home = st.number_input("Warehouse To Home (Distance)", min_value=0.0, max_value=200.0, value=14.0, step=1.0)
    number_of_device = st.number_input("Number Of Devices Registered", min_value=1, max_value=10, value=3)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)

with col2:
    preferred_order_cat = st.selectbox(
        "Preferred Order Category",
        ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )
    number_of_address = st.number_input("Number Of Addresses", min_value=1, max_value=30, value=3)

with col3:
    complain = st.selectbox("Has Complain?", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    days_since_last_order = st.number_input("Days Since Last Order", min_value=0.0, max_value=100.0, value=3.0, step=1.0)
    cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=500.0, value=160.0, step=5.0)

# ---------------------------------------------------------
# 3. Derived Features & Data Preparation
# ---------------------------------------------------------
dissatisfaction_score = complain * (5 - satisfaction_score)
cashback_per_tenure = cashback_amount / (tenure + 1)

input_data = pd.DataFrame([{
    'Tenure': tenure,
    'WarehouseToHome': warehouse_to_home,
    'NumberOfDeviceRegistered': number_of_device,
    'PreferedOrderCat': preferred_order_cat,
    'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_status,
    'NumberOfAddress': number_of_address,
    'Complain': complain,
    'DaySinceLastOrder': days_since_last_order,
    'CashbackAmount': cashback_amount,
    'Dissatisfaction_Score': dissatisfaction_score,
    'Cashback_Per_Tenure': cashback_per_tenure
}])

# ---------------------------------------------------------
# 4. Prediction Execution & Results Output
# ---------------------------------------------------------
st.write("---")

if st.button("🚀 Predict Churn", use_container_width=True):
    selected_pipeline = rf_model if model_choice == "Random Forest Classifier" else ada_model

    prediction = selected_pipeline.predict(input_data)[0]
    probability = selected_pipeline.predict_proba(input_data)[0][1]

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("⚠️ Customer High Risk of Churn")
        else:
            st.success("✅ Customer Retained (Low Churn Risk)")

    with res_col2:
        st.subheader("Churn Probability:")
        st.metric(label="Probability Score", value=f"{probability * 100:.2f}%")
        st.progress(float(probability))