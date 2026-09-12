import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
    page_icon="🛒",
    layout="centered"
)

# =========================================================
# Load artifacts (model pipelines + feature list)
# These .pkl files must sit in the same folder as app.py
# (they are the ones you produced with joblib.dump in the notebook)
# =========================================================
ADA_PATH = "best_ada_pipeline.pkl"
RF_PATH = "best_rf_pipeline.pkl"
FEATURES_PATH = "ecommerce_customer_churn_features.pkl"


@st.cache_resource
def load_artifacts():
    missing = [p for p in [ADA_PATH, RF_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    ada_pipeline = joblib.load(ADA_PATH)
    rf_pipeline = joblib.load(RF_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return ada_pipeline, rf_pipeline, feature_names, []


ada_pipeline, rf_pipeline, feature_names, missing_files = load_artifacts()

st.title("🛒 E-Commerce Customer Churn Prediction")
st.write(
    "This app predicts whether a customer is likely to **churn** "
    "using two trained models: **AdaBoost** and **Random Forest**."
)

if missing_files:
    st.error(
        "Missing required model file(s) in the app folder: "
        + ", ".join(missing_files)
        + ".\n\nMake sure the following files are uploaded next to `app.py`:\n"
        "- `best_ada_pipeline.pkl`\n- `best_rf_pipeline.pkl`\n- `ecommerce_customer_churn_features.pkl`"
    )
    st.stop()

# =========================================================
# Sidebar - model selection
# =========================================================
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Choose the model to use for prediction:",
    ("AdaBoost", "Random Forest", "Compare Both")
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Both models are trained on the same preprocessing pipeline "
    "(KNN imputation + scaling for numeric features, one-hot encoding "
    "for categorical features)."
)

# =========================================================
# Input form - raw features exactly as in the original dataset
# =========================================================
st.header("📋 Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months with the company)", min_value=0.0, max_value=100.0,
        value=10.0, step=1.0
    )
    warehouse_to_home = st.number_input(
        "Warehouse To Home distance (km)", min_value=0.0, max_value=200.0,
        value=15.0, step=1.0
    )
    num_devices = st.number_input(
        "Number of Devices Registered", min_value=1, max_value=10,
        value=4, step=1
    )
    satisfaction_score = st.slider(
        "Satisfaction Score (1 = worst, 5 = best)", min_value=1, max_value=5, value=3
    )
    num_address = st.number_input(
        "Number of Addresses saved", min_value=1, max_value=30,
        value=4, step=1
    )

with col2:
    day_since_last_order = st.number_input(
        "Days Since Last Order", min_value=0.0, max_value=100.0,
        value=5.0, step=1.0
    )
    cashback_amount = st.number_input(
        "Average Cashback Amount", min_value=0.0, max_value=1000.0,
        value=170.0, step=1.0
    )
    complain = st.selectbox(
        "Did the customer file a complaint recently?", ("No", "Yes")
    )
    prefered_order_cat = st.selectbox(
        "Preferred Order Category",
        ("Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others")
    )
    marital_status = st.selectbox(
        "Marital Status", ("Married", "Single", "Divorced")
    )

complain_val = 1 if complain == "Yes" else 0

# =========================================================
# Build the input row with the SAME engineered features
# used during training (Dissatisfaction_Score, Cashback_Per_Tenure)
# =========================================================
input_dict = {
    "Tenure": tenure,
    "WarehouseToHome": warehouse_to_home,
    "NumberOfDeviceRegistered": num_devices,
    "PreferedOrderCat": prefered_order_cat,
    "SatisfactionScore": satisfaction_score,
    "MaritalStatus": marital_status,
    "NumberOfAddress": num_address,
    "Complain": complain_val,
    "DaySinceLastOrder": day_since_last_order,
    "CashbackAmount": cashback_amount,
}

# Feature engineering identical to the notebook
input_dict["Dissatisfaction_Score"] = complain_val * (5 - satisfaction_score)
input_dict["Cashback_Per_Tenure"] = cashback_amount / (tenure + 1)

input_df = pd.DataFrame([input_dict])

# Reorder columns to match the training feature order, if available
try:
    input_df = input_df[feature_names]
except Exception:
    st.warning(
        "Could not reorder columns to match the training feature list exactly; "
        "proceeding with the columns as built. If this causes an error, check "
        "that the feature names above match `ecommerce_customer_churn_features.pkl`."
    )

with st.expander("🔍 Show the exact row sent to the model"):
    st.dataframe(input_df)

# =========================================================
# Prediction
# =========================================================
st.header("🔮 Prediction")

if st.button("Predict Churn"):
    try:
        results = {}

        if model_choice in ("AdaBoost", "Compare Both"):
            ada_pred = ada_pipeline.predict(input_df)[0]
            ada_proba = ada_pipeline.predict_proba(input_df)[0][1]
            results["AdaBoost"] = (ada_pred, ada_proba)

        if model_choice in ("Random Forest", "Compare Both"):
            rf_pred = rf_pipeline.predict(input_df)[0]
            rf_proba = rf_pipeline.predict_proba(input_df)[0][1]
            results["Random Forest"] = (rf_pred, rf_proba)

        if len(results) == 1:
            model_name, (pred, proba) = list(results.items())[0]
            if pred == 1:
                st.error(f"⚠️ {model_name}: This customer is **likely to churn**.")
            else:
                st.success(f"✅ {model_name}: This customer is **not likely to churn**.")
            st.metric("Churn Probability", f"{proba * 100:.2f}%")
            st.progress(min(max(proba, 0.0), 1.0))
        else:
            cols = st.columns(len(results))
            for c, (model_name, (pred, proba)) in zip(cols, results.items()):
                with c:
                    st.subheader(model_name)
                    if pred == 1:
                        st.error("Likely to churn ⚠️")
                    else:
                        st.success("Not likely to churn ✅")
                    st.metric("Probability", f"{proba * 100:.2f}%")

    except Exception as e:
        st.exception(e)

st.markdown("---")
st.caption(
    "Built with Streamlit • Models: AdaBoost & Random Forest • "
    "Preprocessing: KNN Imputer + StandardScaler + OneHotEncoder"
)
