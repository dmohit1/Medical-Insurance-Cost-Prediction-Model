
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------
# App configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Estimator",
    layout="wide"
)

MODEL_PATH = "insurance_model.pkl"
FEATURES_PATH = "model_features.pkl"

REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000


# --------------------------------------------------
# Load trained model and feature names
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None

    if not os.path.exists(FEATURES_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    return model, feature_cols


# --------------------------------------------------
# BMI classification
# --------------------------------------------------
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# --------------------------------------------------
# Prepare model input
# --------------------------------------------------
def build_input_dataframe(
    age, sex, bmi, children, smoker, region, feature_cols
):
    category = bmi_category(bmi)

    data = {
        "age": [age],
        "sex_male": [1 if sex == "male" else 0],
        "bmi": [bmi],
        "children": [children],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
        "bmi_category_obese": [1 if category == "Obese" else 0],
        "bmi_category_Underweight": [
            1 if category == "Underweight" else 0
        ],
        "bmi_category_Overweight": [
            1 if category == "Overweight" else 0
        ],
    }

    encoded = pd.DataFrame(data)

    # Match the exact feature order used during training
    return encoded.reindex(columns=feature_cols, fill_value=0)


# --------------------------------------------------
# Load artifacts and render app
# --------------------------------------------------
model, feature_cols = load_artifacts()

st.title("Insurance Cost Estimator")
st.caption("Estimate annual insurance charges based on customer details.")

if model is None or feature_cols is None:
    st.error(
        "Model files not found. Ensure insurance_model.pkl "
        "and model_features.pkl are in the app directory."
    )
    st.stop()


left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Patient Details")

    age = st.slider("Age", 18, 100, 30)
    sex = st.radio("Sex", ["male", "female"])
    bmi = st.slider("BMI", 15.0, 55.0, 26.0, 0.1)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["yes", "no"], horizontal=True)

    region = st.selectbox(
        "Region",
        ["northwest", "northeast", "southwest", "southeast"]
    )

    predict_clicked = st.button(
        "Estimate charges",
        type="primary",
        use_container_width=True
    )


with right:
    st.subheader("Estimated Charges")

    if predict_clicked:
        input_row = build_input_dataframe(
            age, sex, bmi, children, smoker, region, feature_cols
        )

        try:
            prediction = float(model.predict(input_row)[0])

            # Use this only if the model was trained on log1p(charges).
            pred_charges = np.expm1(prediction)

            if not np.isfinite(pred_charges) or pred_charges < 0:
                st.error("The model returned an invalid prediction.")
                st.stop()

            st.metric(
                "Predicted annual charge",
                f"${pred_charges:,.2f}"
            )

            st.divider()

            compare_df = pd.DataFrame({
                "Group": [
                    "Your Prediction",
                    "Non-Smoker Average",
                    "Smoker Average"
                ],
                "Charges": [
                    pred_charges,
                    REF_AVG_NON_SMOKER,
                    REF_AVG_SMOKER
                ]
            }).set_index("Group")

            st.subheader("Charges Comparison")

            st.bar_chart(
                compare_df,
                y="Charges",
                use_container_width=True
            )

            st.dataframe(
                compare_df,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("Enter the details and click Estimate charges.")
