import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
import warnings

# Configure warnings and matplotlib
warnings.filterwarnings('ignore', category=UserWarning)
plt.set_loglevel('WARNING')  # This replaces the deprecated Streamlit option

# Streamlit page configuration
st.set_page_config(
    page_title="📈 E-commerce Readiness Prediction Dashboard",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    """Load all required models and data"""
    artifacts = {
        "best_cart": joblib.load("best_cart_model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "X_train": pd.read_csv("X_train.csv"),
        "feature_names": list(pd.read_csv("feature_names.csv").squeeze())
    }
    
    # Ensure feature names match between all components
    if hasattr(artifacts["scaler"], 'feature_names_in_'):
        artifacts["feature_names"] = list(artifacts["scaler"].feature_names_in_)
    
    return artifacts

try:
    artifacts = load_artifacts()
    
    # Initialize explainers
    explainer = shap.Explainer(artifacts["best_cart"], artifacts["X_train"])
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=artifacts["X_train"].values,
        mode='regression',
        feature_names=artifacts["feature_names"],
        verbose=False
    )

    # UI Components
    st.title("📈 E-commerce Readiness Prediction Dashboard")
    
    # Sidebar inputs with proper feature names
    st.sidebar.header("Input Features")
    user_input = []
    for col in artifacts["feature_names"]:
        user_input.append(st.sidebar.number_input(col, value=0.0))
    
    # Create DataFrame with correct feature names
    user_df = pd.DataFrame([user_input], columns=artifacts["feature_names"])
    
    # Scale the input (with feature name preservation)
    try:
        user_scaled = artifacts["scaler"].transform(user_df)
        user_scaled = pd.DataFrame(user_scaled, columns=artifacts["feature_names"])
    except Exception as e:
        st.error(f"Scaling error: {str(e)}")
        st.stop()

    # Prediction
    prediction = artifacts["best_cart"].predict(user_scaled)[0]
    st.subheader("🎯 Predicted Ecommerce Growth")
    st.metric("Prediction", f"{round(prediction, 2)}")

    # SHAP Explanation
    st.subheader("SHAP Explanation")
    try:
        # Calculate SHAP values
        shap_values = explainer(user_scaled)
        
        # Create waterfall plot
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"SHAP visualization failed: {str(e)}")

    # LIME Explanation
    st.subheader("🧠 LIME Explanation")
    try:
        lime_exp = lime_explainer.explain_instance(
            data_row=user_scaled.values[0],
            predict_fn=artifacts["best_cart"].predict,
            num_features=len(artifacts["feature_names"])
        )
        st.components.v1.html(lime_exp.as_html(), height=800)
    except Exception as e:
        st.error(f"LIME explanation failed: {str(e)}")

except FileNotFoundError as e:
    st.error(f"Missing required file: {str(e)}")
except Exception as e:
    st.error(f"Application error: {str(e)}")