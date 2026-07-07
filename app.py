import streamlit as st
import joblib
import warnings
import numpy as np
import pandas as pd
import time
from sklearn.exceptions import InconsistentVersionWarning

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────
#  LOAD MODEL  (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        return joblib.load("randomclassifiermodel.pkl")

model = load_model()

# ─────────────────────────────────────────
#  GLOBAL CSS  — clean, professional dark SaaS theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Hide Streamlit chrome (top white bar / menu / footer) ── */
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
    visibility: hidden;
    height: 0;
}
div.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #dbe4f3;
}
.stApp {
    background-color: #0a0f1e;
    background-image:
        radial-gradient(circle at 8% 8%, rgba(115,103,240,0.18), transparent 30%),
        radial-gradient(circle at 92% 15%, rgba(79,172,254,0.14), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(0,224,160,0.08), transparent 35%),
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(160deg, #0a0f1e 0%, #0d1326 55%, #0a1428 100%);
    background-size: auto, auto, auto, 42px 42px, 42px 42px, auto;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* ── Hero header ──────────────────────── */
.hero-header {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,224,160,0.1);
    border: 1px solid rgba(0,224,160,0.35);
    color: #00e0a0;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}
.hero-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00e0a0;
    display: inline-block;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4facfe, #7367f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7c8aa8;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Divider ──────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(115,103,240,0.4), transparent);
    border: none;
    margin: 1.2rem 0 1.6rem;
}

/* ── Section label ────────────────────── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: #7367f0;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Native Streamlit bordered container = card ──────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.02);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: 1px solid rgba(115,103,240,0.2) !important;
    border-radius: 16px !important;
    background: rgba(20,26,48,0.6) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    border-color: rgba(115,103,240,0.55) !important;
    box-shadow: 0 6px 22px -10px rgba(115,103,240,0.4);
}

/* ── Card heading text ────────────────── */
.card-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.98rem;
    color: #dbe4f3;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px dashed rgba(115,103,240,0.25);
}

/* ── Prediction result cards ─────────── */
.result-card-churn {
    background: linear-gradient(135deg, rgba(255,60,90,0.12), rgba(120,0,60,0.18));
    border: 1px solid rgba(255,80,110,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-card-safe {
    background: linear-gradient(135deg, rgba(0,200,140,0.1), rgba(0,80,60,0.18));
    border: 1px solid rgba(0,220,150,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-icon  { font-size: 3rem; margin-bottom: 0.4rem; }
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-label-churn { color: #ff5c7a; }
.result-label-safe  { color: #00e0a0; }
.result-prob { font-size: 1rem; color: #a0b0cc; }

/* ── Probability bar ─────────────────── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    overflow: hidden;
    height: 10px;
    margin: 0.6rem 0 0.2rem;
}
.prob-bar-fill-churn {
    height: 100%;
    background: linear-gradient(90deg, #ff3d63, #ff9d5c);
    border-radius: 999px;
}
.prob-bar-fill-safe {
    height: 100%;
    background: linear-gradient(90deg, #00c8ff, #00e0a0);
    border-radius: 999px;
}

/* ── Risk signal cards ────────────────── */
.risk-card {
    border: 1px solid rgba(255,170,0,0.28);
    border-radius: 12px;
    padding: 1rem;
    background: rgba(255,170,0,0.04);
    height: 100%;
}
.risk-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #ffb347;
    margin-top: 0.3rem;
}
.risk-desc { font-size: 0.82rem; color: #8b96ad; margin-top: 0.3rem; }

.ok-card {
    border: 1px solid rgba(0,224,160,0.25);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    background: rgba(0,224,160,0.04);
}

/* ── Sidebar ──────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0b1024 !important;
    border-right: 1px solid rgba(115,103,240,0.15);
}
section[data-testid="stSidebar"] .stMarkdown p { color: #7c8aa8; }

/* ── Streamlit widget labels ──────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #9aa8c7 !important;
    font-size: 0.85rem !important;
    font-weight: 500;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(115,103,240,0.35) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}
div[data-baseweb="select"] > div:hover {
    border-color: #7367f0 !important;
}
/* Force readable text color for the selected value shown in the box */
div[data-baseweb="select"] div, div[data-baseweb="select"] span {
    color: #f0f3fc !important;
}
/* Dropdown menu (renders in a portal) — dark bg + light readable text */
ul[data-baseweb="menu"], div[data-baseweb="popover"] ul {
    background-color: #141a30 !important;
    border: 1px solid rgba(115,103,240,0.3) !important;
}
li[role="option"] {
    background-color: transparent !important;
    color: #dbe4f3 !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: rgba(115,103,240,0.25) !important;
    color: #ffffff !important;
}

/* ── Sliders ──────────────────────────── */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #4facfe, #7367f0) !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #7c8aa8 !important;
}
div[data-testid="stThumbValue"] {
    color: #f0f3fc !important;
    font-weight: 600 !important;
}

/* ── Number inputs ────────────────────── */
div[data-testid="stNumberInputContainer"][data-testid="stNumberInputContainer"] div[data-baseweb="input"][data-baseweb="input"] {
    background: #141a30 !important;
    border: 1px solid rgba(115,103,240,0.4) !important;
    border-radius: 10px !important;
}
div[data-testid="stNumberInputContainer"][data-testid="stNumberInputContainer"] div[data-baseweb="base-input"][data-baseweb="base-input"] {
    background: #141a30 !important;
}
input[data-testid="stNumberInputField"][data-testid="stNumberInputField"] {
    background: #141a30 !important;
    color: #f0f3fc !important;
    -webkit-text-fill-color: #f0f3fc !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
button[data-testid="stNumberInputStepDown"][data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"][data-testid="stNumberInputStepUp"] {
    background: #1a2140 !important;
    border-color: rgba(115,103,240,0.4) !important;
}
button[data-testid="stNumberInputStepDown"] svg,
button[data-testid="stNumberInputStepUp"] svg {
    fill: #f0f3fc !important;
    color: #f0f3fc !important;
}

/* ── Radio buttons (used for quick Yes/No style choices) ── */
div[role="radiogroup"] {
    gap: 1rem;
}
div[role="radiogroup"] input {
    accent-color: #7367f0;
}
div[role="radiogroup"] label p {
    color: #b8c4e0 !important;
    font-size: 0.85rem !important;
}

/* ── Submit button ────────────────────── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    background: linear-gradient(90deg, #4facfe, #7367f0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.88; }

/* ── Metric tiles in sidebar ──────────── */
div[data-testid="stMetric"] {
    background: rgba(115,103,240,0.06) !important;
    border: 1px solid rgba(115,103,240,0.15) !important;
    border-radius: 10px !important;
    padding: 0.7rem !important;
}
div[data-testid="stMetric"] label { color: #7c8aa8 !important; font-size: 0.72rem !important; }
div[data-testid="stMetricValue"] { color: #7367f0 !important; font-family: 'Space Grotesk', sans-serif !important; }

.footer-text {
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    color: #3a4568;
    padding: 1.5rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge"><span class="dot"></span> Model Live · Real-time Predictions</div>
  <div class="hero-title">⚡ ChurnGuard AI</div>
  <div class="hero-sub">Telecom Customer Churn Intelligence Platform</div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────
st.markdown('<div class="section-label">◈ Customer Profile Input</div>', unsafe_allow_html=True)

with st.form("prediction_form"):

    # ── Row 1 : demographics / account / billing ──────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="card-heading">👤 Demographics</div>', unsafe_allow_html=True)
            senior      = st.radio("Senior Citizen",  ["No", "Yes"], horizontal=True)
            gender      = st.radio("Gender",           ["Male", "Female"], horizontal=True)
            partner     = st.radio("Partner",          ["Yes", "No"], horizontal=True)
            dependents  = st.radio("Dependents",       ["No", "Yes"], horizontal=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="card-heading">📞 Account Details</div>', unsafe_allow_html=True)
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, step=10.0)
            contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col3:
        with st.container(border=True):
            st.markdown('<div class="card-heading">💳 Billing</div>', unsafe_allow_html=True)
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
            payment_method    = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            phone_service  = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # ── Row 2 : internet & add-ons ────────
    col4, col5, col6 = st.columns(3, gap="medium")

    with col4:
        with st.container(border=True):
            st.markdown('<div class="card-heading">🌐 Internet</div>', unsafe_allow_html=True)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
            online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])

    with col5:
        with st.container(border=True):
            st.markdown('<div class="card-heading">🔒 Protection & Support</div>', unsafe_allow_html=True)
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])

    with col6:
        with st.container(border=True):
            st.markdown('<div class="card-heading">📺 Streaming</div>', unsafe_allow_html=True)
            streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # ── Submit ────────────────────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡  Run Churn Analysis")


# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(
    senior, gender, partner, dependents, tenure,
    monthly_charges, total_charges, phone_service,
    multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract,
    paperless_billing, payment_method
):
    feat = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Female":  1 if gender == "Female" else 0,
        "gender_Male":    1 if gender == "Male"   else 0,
        "Partner_No":  1 if partner == "No"  else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_No":  1 if dependents == "No"  else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_No":  1 if phone_service == "No"  else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_No":               1 if multiple_lines == "No"               else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes":              1 if multiple_lines == "Yes"              else 0,
        "InternetService_DSL":         1 if internet_service == "DSL"         else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No":          1 if internet_service == "No"          else 0,
        "OnlineSecurity_No":                   1 if online_security == "No"                   else 0,
        "OnlineSecurity_No internet service":  1 if online_security == "No internet service"  else 0,
        "OnlineSecurity_Yes":                  1 if online_security == "Yes"                  else 0,
        "OnlineBackup_No":                     1 if online_backup == "No"                     else 0,
        "OnlineBackup_No internet service":    1 if online_backup == "No internet service"    else 0,
        "OnlineBackup_Yes":                    1 if online_backup == "Yes"                    else 0,
        "DeviceProtection_No":                 1 if device_protection == "No"                 else 0,
        "DeviceProtection_No internet service":1 if device_protection == "No internet service"else 0,
        "DeviceProtection_Yes":                1 if device_protection == "Yes"                else 0,
        "TechSupport_No":                      1 if tech_support == "No"                      else 0,
        "TechSupport_No internet service":     1 if tech_support == "No internet service"     else 0,
        "TechSupport_Yes":                     1 if tech_support == "Yes"                     else 0,
        "StreamingTV_No":                      1 if streaming_tv == "No"                      else 0,
        "StreamingTV_No internet service":     1 if streaming_tv == "No internet service"     else 0,
        "StreamingTV_Yes":                     1 if streaming_tv == "Yes"                     else 0,
        "StreamingMovies_No":                  1 if streaming_movies == "No"                  else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes":                 1 if streaming_movies == "Yes"                 else 0,
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year":       1 if contract == "One year"       else 0,
        "Contract_Two year":       1 if contract == "Two year"       else 0,
        "PaperlessBilling_No":  1 if paperless_billing == "No"  else 0,
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":   1 if payment_method == "Credit card (automatic)"   else 0,
        "PaymentMethod_Electronic check":          1 if payment_method == "Electronic check"          else 0,
        "PaymentMethod_Mailed check":              1 if payment_method == "Mailed check"              else 0,
    }
    return pd.DataFrame([feat])


# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────
if submitted:
    with st.spinner("Analyzing customer profile..."):
        time.sleep(0.6)

    X = build_features(
        senior, gender, partner, dependents, tenure,
        monthly_charges, total_charges, phone_service,
        multiple_lines, internet_service, online_security,
        online_backup, device_protection, tech_support,
        streaming_tv, streaming_movies, contract,
        paperless_billing, payment_method
    )

    pred       = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    churn_prob = proba[1]
    safe_prob  = proba[0]

    churn_pct  = round(churn_prob * 100, 1)
    safe_pct   = round(safe_prob  * 100, 1)
    will_churn = (pred == 1) or (str(pred).lower() == "yes")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">◈ Prediction Result</div>', unsafe_allow_html=True)

    if will_churn:
        st.markdown(f"""
        <div class="result-card-churn">
          <div class="result-icon">🚨</div>
          <div class="result-label result-label-churn">High Churn Risk</div>
          <div class="result-prob">Churn Probability: <strong>{churn_pct}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-safe">
          <div class="result-icon">✅</div>
          <div class="result-label result-label-safe">Customer Retained</div>
          <div class="result-prob">Retention Probability: <strong>{safe_pct}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(border=True):
            st.markdown(f"""
            <div class="section-label" style="margin-bottom:0.4rem;">⚠ Churn Probability</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; color:#ff5c7a;">{churn_pct}%</div>
            <div class="prob-bar-wrap">
              <div class="prob-bar-fill-churn" style="width:{churn_pct}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        with st.container(border=True):
            st.markdown(f"""
            <div class="section-label" style="margin-bottom:0.4rem;">✔ Retention Probability</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; color:#00e0a0;">{safe_pct}%</div>
            <div class="prob-bar-wrap">
              <div class="prob-bar-fill-safe" style="width:{safe_pct}%;"></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Key risk factors ──────────────────
    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">◈ Key Risk Signals</div>', unsafe_allow_html=True)

    risks = []
    if contract == "Month-to-month":
        risks.append(("📋", "Contract Risk",   "Month-to-month contracts are the #1 churn driver"))
    if internet_service == "Fiber optic" and tech_support == "No":
        risks.append(("🔧", "No Tech Support", "Fiber customers without support churn 2× faster"))
    if tenure < 12:
        risks.append(("⏳", "New Customer",     f"Tenure only {tenure} months — high vulnerability"))
    if monthly_charges > 80:
        risks.append(("💸", "High Billing",     f"${monthly_charges}/mo is above average threshold"))
    if payment_method == "Electronic check":
        risks.append(("💳", "Payment Method",  "Electronic check users have higher churn rate"))

    if risks:
        rcols = st.columns(len(risks), gap="small")
        for i, (icon, title, desc) in enumerate(risks):
            with rcols[i]:
                st.markdown(f"""
                <div class="risk-card">
                  <div style="font-size:1.3rem;">{icon}</div>
                  <div class="risk-title">{title}</div>
                  <div class="risk-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ok-card">
          <span style="color:#00e0a0; font-family:'Space Grotesk',sans-serif; font-size:0.9rem;">
            ✓ No major risk signals detected — customer profile looks healthy
          </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
<hr class="divider" style="margin-top:2.5rem;">
<div class="footer-text">
  CHURNGUARD AI · RANDOM FOREST CLASSIFIER · 45 FEATURES · BINARY CLASSIFICATION · BY AYMAN
</div>
""", unsafe_allow_html=True)