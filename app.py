"""
app.py
An interactive, visually polished Streamlit app that predicts churn
for a single customer based on their details.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")

# ---- Custom styling ----
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background-color: #1E2530;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #2D3646;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #E63E3E, #E65555);
    }
</style>
""", unsafe_allow_html=True)

model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")
columns = joblib.load("models/columns.pkl")

# ---- Header ----
st.markdown('<p class="main-title">📊 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by a Random Forest model trained on 7,000+ telecom customer records</p>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

user_input = {}

with left:
    st.markdown("#### 👤 Customer Profile")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            user_input["gender"] = st.selectbox("Gender", ["Male", "Female"])
            user_input["Partner"] = st.selectbox("Has Partner", ["Yes", "No"])
        with c2:
            user_input["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
            user_input["Dependents"] = st.selectbox("Has Dependents", ["Yes", "No"])

    st.markdown("#### 📄 Contract Details")
    user_input["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    user_input["Contract"] = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    user_input["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])

with right:
    st.markdown("#### 💳 Billing & Service")
    user_input["MonthlyCharges"] = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
    user_input["TotalCharges"] = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)
    user_input["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    user_input["PaymentMethod"] = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

# Fill remaining columns the model expects with sensible defaults
defaults = {
    "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
}
for col in columns:
    if col not in user_input and col in defaults:
        user_input[col] = defaults[col]

st.markdown("---")
predict_clicked = st.button("🔮 Predict Churn")

if predict_clicked:
    input_df = pd.DataFrame([user_input])

    for col, le in encoders.items():
        if col in input_df.columns and col != "Churn":
            input_df[col] = le.transform(input_df[col].astype(str))

    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result_col, gauge_col = st.columns([1, 1], gap="large")

    with result_col:
        if prediction == 1:
            st.markdown(f"""
            <div style="background-color:#3D1A1A; border-left: 6px solid #FF4B4B; padding: 1.5rem; border-radius: 10px;">
                <h3 style="color:#FF6B6B; margin:0;">⚠️ Likely to Churn</h3>
                <p style="color:#E0E0E0; margin-top:0.5rem;">This customer shows a <b>{probability:.1%}</b> probability of cancelling their subscription.</p>
                <p style="color:#B0B0B0; font-size:0.9rem;">Consider a retention offer, contract upgrade incentive, or proactive support outreach.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#1A3D24; border-left: 6px solid #2ECC71; padding: 1.5rem; border-radius: 10px;">
                <h3 style="color:#2ECC71; margin:0;">✅ Likely to Stay</h3>
                <p style="color:#E0E0E0; margin-top:0.5rem;">This customer shows only a <b>{probability:.1%}</b> probability of churning.</p>
                <p style="color:#B0B0B0; font-size:0.9rem;">A healthy, low-risk customer profile.</p>
            </div>
            """, unsafe_allow_html=True)

    with gauge_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 40}},
            title={'text': "Churn Risk", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#9CA3AF"},
                'bar': {'color': "#FF4B4B" if prediction == 1 else "#2ECC71"},
                'bgcolor': "#1E2530",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': '#1A3D24'},
                    {'range': [40, 70], 'color': '#3D3A1A'},
                    {'range': [70, 100], 'color': '#3D1A1A'},
                ],
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                           paper_bgcolor="rgba(0,0,0,0)", font={'color': "#E0E0E0"})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Adjust the customer details above, then click **Predict Churn** to see the result.")