import streamlit as st
import pandas as pd
import joblib

# --- 1. Page Configuration ---
st.set_page_config(page_title="Machine Foundation Predictor", page_icon="🏗️", layout="centered")

# --- 2. Load the Model ---
# The @st.cache_resource decorator keeps the model in memory so it doesn't reload every time
@st.cache_resource
def load_model():
    return joblib.load('xgboost_foundation_model.pkl')

model = load_model()

# --- 3. App Header ---
st.title("🏗️ Machine Foundation Vibration Predictor")
st.markdown("""
This app uses an **XGBoost Machine Learning model** to predict the dynamic displacement amplitude ($z$) of a machine foundation based on soil and structural parameters.
""")
st.divider()

# --- 4. Sidebar for User Inputs ---
st.sidebar.header("⚙️ Engineering Parameters")
st.sidebar.markdown("Adjust the sliders to simulate different foundation conditions.")

A = st.sidebar.slider("Base Area, A (m²)", 0.30, 0.45, 0.35, step=0.01)
F_s = st.sidebar.slider("Shape Coefficient, F_s", 3.50, 4.30, 3.95, step=0.01)
m = st.sidebar.slider("Total Mass, m (kg)", 330.0, 890.0, 600.0, step=10.0)
k_mega = st.sidebar.slider("Soil Stiffness, k (MN/m)", 7.9, 10.9, 9.5, step=0.1)
m_e_e = st.sidebar.slider("Eccentric Force, m_e·e (N·s²)", 0.015, 0.175, 0.090, step=0.005)
f = st.sidebar.slider("Operating Frequency, f (Hz)", 5.0, 50.0, 30.0, step=0.5)

# Convert stiffness from MN/m to N/m for the model
k = k_mega * 1e6

# --- 5. Main Layout ---
st.subheader("Current Input Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Mass (m)", f"{m} kg")
col2.metric("Stiffness (k)", f"{k_mega} MN/m")
col3.metric("Frequency (f)", f"{f} Hz")

st.write("") # Spacer

# --- 6. Prediction Logic ---
if st.button("Predict Displacement (z)", type="primary"):
    # Format inputs exactly as the model expects them
    input_df = pd.DataFrame({
        'A': [A],
        'F_s': [F_s],
        'm': [m],
        'k': [k],
        'm_e_e': [m_e_e],
        'f': [f]
    })
    
    # Generate Prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.success(f"### Predicted Amplitude (z): {prediction:.2f} × 10⁻⁶ m")
    
    # Contextual Engineering Warnings
    if prediction > 800:
        st.error("⚠️ **Critical Vibration Warning:** Amplitude is exceedingly high. Severe risk of resonance. Consider altering operating frequency or increasing mass/stiffness.")
    elif prediction > 400:
        st.warning("⚠️ **Moderate Vibration:** Amplitude is elevated. Verify against acceptable operational limits for the specific machinery.")
    else:
        st.info("✅ **Safe Operation:** Vibration amplitude is relatively low and likely within safe design limits.")