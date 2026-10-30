import math
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components  # fallback for older Streamlit versions

st.set_page_config(page_title="ChurnSense | Churn Risk Predictor", layout="wide", page_icon="📡")

# =====================================================================
# 1. STYLING: fonts, animated background, glass cards, animated widgets
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, .display { font-family: 'Sora', sans-serif !important; letter-spacing: -0.5px; }

/* ---------- animated background ---------- */
.stApp { background: #060a18; }
.stApp::before {
    content: ""; position: fixed; inset: -10%; z-index: 0; pointer-events: none;
    background:
        radial-gradient(620px 420px at 12% 8%,  rgba(99,102,241,.38), transparent 60%),
        radial-gradient(520px 420px at 88% 18%, rgba(236,72,153,.26), transparent 60%),
        radial-gradient(720px 520px at 50% 105%, rgba(20,184,166,.24), transparent 60%);
    animation: drift 16s ease-in-out infinite alternate;
}
.stApp::after {
    content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(ellipse at 50% 30%, black 20%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 50% 30%, black 20%, transparent 75%);
}
@keyframes drift { from { transform: translate3d(0,0,0) scale(1); } to { transform: translate3d(0,-40px,0) scale(1.1); } }
.block-container { position: relative; z-index: 1; padding-top: 2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---------- hero ---------- */
.eyebrow { font-size: .75rem; letter-spacing: 4px; color: #818cf8; font-weight: 600; margin-bottom: .2rem; }
.title {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 3.6rem; line-height: 1.05; margin: 0;
    background: linear-gradient(90deg, #a5b4fc, #f0abfc, #5eead4, #a5b4fc);
    background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: shine 8s linear infinite;
}
@keyframes shine { to { background-position: 300% 0; } }
.subtitle { color: #94a3b8; font-size: 1.05rem; margin: .4rem 0 1rem 0; max-width: 640px; }
.chips { display: flex; gap: .6rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.chip {
    padding: .35rem .85rem; border-radius: 999px; font-size: .78rem; color: #cbd5e1;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1);
    backdrop-filter: blur(6px); animation: rise .8s ease both;
}
.chip b { color: #5eead4; }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }

/* ---------- glass cards (st.container with key="card_*") ---------- */
[class*="st-key-card"] {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.09);
    border-radius: 18px; padding: 1.2rem 1.4rem; backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,.35); animation: rise .7s ease both;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
[class*="st-key-card"]:hover {
    transform: translateY(-3px); border-color: rgba(129,140,248,.5);
    box-shadow: 0 14px 40px rgba(99,102,241,.22);
}
.card-title { font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1rem; color: #e2e8f0; margin-bottom: .2rem; }
.card-title span { color: #818cf8; margin-right: .4rem; }

/* ---------- widgets ---------- */
[data-testid="stWidgetLabel"] p { color: #94a3b8; font-size: .85rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #94a3b8 !important; }
div[data-baseweb="select"] > div { background: rgba(255,255,255,.05) !important; border-color: rgba(255,255,255,.12) !important; border-radius: 10px; }
button[data-baseweb="tab"] { font-family: 'Sora', sans-serif; font-weight: 600; }

.stButton > button {
    position: relative; overflow: hidden; border: none; border-radius: 14px; padding: .8rem 1rem;
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; letter-spacing: .5px; color: white;
    background: linear-gradient(90deg, #6366f1, #d946ef, #14b8a6); background-size: 200% 100%;
    animation: flow 6s ease infinite; box-shadow: 0 0 28px rgba(139,92,246,.45);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-2px) scale(1.01); box-shadow: 0 0 42px rgba(217,70,239,.6); color: white; }
.stButton > button::after {
    content: ""; position: absolute; top: 0; left: -75%; width: 50%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,.35), transparent);
    transform: skewX(-20deg); animation: sweep 3.2s ease-in-out infinite;
}
@keyframes flow { 0%,100% { background-position: 0 0; } 50% { background-position: 100% 0; } }
@keyframes sweep { 0% { left: -75%; } 60%,100% { left: 130%; } }

/* ---------- result pieces ---------- */
.verdict { padding: 1.3rem 1.5rem; border-radius: 16px; animation: rise .7s ease both; }
.verdict h3 { margin: 0 0 .35rem 0; font-size: 1.5rem; }
.verdict p { margin: 0; color: #cbd5e1; font-size: .95rem; }
.flags { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem; }
.flag { padding: .28rem .7rem; border-radius: 8px; font-size: .78rem; background: rgba(244,63,94,.12);
        border: 1px solid rgba(244,63,94,.35); color: #fda4af; }
.flag.ok { background: rgba(20,184,166,.12); border-color: rgba(20,184,166,.35); color: #5eead4; }

.lever { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: .85rem 1rem;
         margin-bottom: .6rem; border-radius: 12px; background: rgba(255,255,255,.04);
         border: 1px solid rgba(255,255,255,.08); animation: rise .7s ease both; }
.lever .name { color: #e2e8f0; font-weight: 500; font-size: .92rem; }
.lever .shift { font-family: 'Sora', sans-serif; font-weight: 700; color: #5eead4; white-space: nowrap; }
.lever .shift small { color: #94a3b8; font-weight: 400; }
.note { color: #64748b; font-size: .75rem; margin-top: .4rem; }

.seg { padding: 1rem 1.2rem; border-radius: 14px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); }
.seg.hot { border-color: rgba(244,63,94,.55); box-shadow: 0 0 24px rgba(244,63,94,.18); }
.seg .big { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2rem; color: #f8fafc; }
.seg .meta { color: #94a3b8; font-size: .82rem; }

.footer-wrapper { display: flex; justify-content: center; margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid rgba(255,255,255,.08); }
.footer-credit { padding: .55rem 1.3rem; border-radius: 999px; font-size: .9rem; color: #94a3b8;
                 background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.1); transition: .3s ease; }
.footer-credit:hover { border-color: #818cf8; box-shadow: 0 0 20px rgba(129,140,248,.3); }
.footer-credit a { color: #a5b4fc; text-decoration: none; font-weight: 600; }
.footer-credit a:hover { color: #f0abfc; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. MODEL LOADING + PREDICTION HELPERS
# =====================================================================
@st.cache_resource
def load_artifacts():
    return (joblib.load("models/churn_model.pkl"),
            joblib.load("models/encoders.pkl"),
            joblib.load("models/columns.pkl"))


model, encoders, columns = load_artifacts()


def predict_proba(customer: dict) -> float:
    """Encode one customer dict exactly like train_model.py and return P(churn)."""
    df = pd.DataFrame([customer])
    for col, le in encoders.items():
        if col in df.columns and col != "Churn":
            df[col] = le.transform(df[col].astype(str))
    return float(model.predict_proba(df[columns])[0][1])


def render_html(html: str, height: int):
    """Render our own trusted HTML/JS in an iframe (st.iframe on new Streamlit, components.html on old)."""
    if hasattr(st, "iframe"):
        st.iframe(html.strip(), height=height)
    else:
        components.html(html, height=height)


def yn(flag: bool) -> str:
    return "Yes" if flag else "No"


TIERS = [  # (upper bound, label, colour 1, colour 2, headline, advice)
    (0.35, "LOW RISK", "#22d3a6", "#38bdf8", "Likely to stay",
     "A healthy, low-risk profile. No intervention needed beyond routine service."),
    (0.50, "MODERATE", "#facc15", "#fb923c", "Watch closely",
     "Below the churn threshold but trending up. Light-touch engagement is worthwhile."),
    (0.70, "HIGH RISK", "#fb923c", "#f43f5e", "Likely to churn",
     "This customer is more likely than not to leave. Prioritise a retention offer."),
    (1.01, "CRITICAL", "#f43f5e", "#c026d3", "Very likely to churn",
     "Very high churn probability. Escalate to proactive retention outreach."),
]


def tier_for(p: float):
    return next(t for t in TIERS if p < t[0])


def gauge_html(prob: float) -> str:
    """Animated ring gauge with count-up number (pure SVG + JS, no extra libraries)."""
    _, label, c1, c2, *_ = tier_for(prob)
    circ = 2 * math.pi * 115
    arc = circ * 0.75
    html = """
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@600;800&family=Inter:wght@500;600&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;background:transparent;font-family:'Inter',sans-serif;overflow:hidden}
  .wrap{position:relative;width:310px;height:310px;margin:0 auto;animation:pop .9s cubic-bezier(.2,1.2,.3,1) both}
  .wrap.hot{animation:pop .9s cubic-bezier(.2,1.2,.3,1) both, pulse 2.4s ease-in-out 1s infinite}
  svg{width:310px;height:310px;overflow:visible}
  .spin{transform-origin:150px 150px;animation:spin 50s linear infinite}
  .spin2{transform-origin:150px 150px;animation:spin 30s linear infinite reverse}
  #arc{transition:stroke-dasharray 1.8s cubic-bezier(.22,1,.36,1)}
  .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .cap{font-size:10px;letter-spacing:4px;color:#94a3b8;font-weight:600}
  .num{font-family:'Sora',sans-serif;font-weight:800;font-size:68px;color:#fff;letter-spacing:-3px;line-height:1.05;
       text-shadow:0 0 28px %%C1%%88}
  .num small{font-size:26px;letter-spacing:0;opacity:.65;margin-left:2px}
  .pill{margin-top:8px;padding:5px 16px;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:2.5px;color:#fff;
        background:linear-gradient(90deg,%%C1%%,%%C2%%);box-shadow:0 0 22px %%C1%%99}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pop{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}
  @keyframes pulse{0%,100%{filter:drop-shadow(0 0 6px %%C1%%55)}50%{filter:drop-shadow(0 0 22px %%C1%%cc)}}
</style>
<div class="wrap %%HOT%%">
  <svg viewBox="0 0 300 300">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="%%C1%%"/><stop offset="1" stop-color="%%C2%%"/>
      </linearGradient>
      <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <circle class="spin"  cx="150" cy="150" r="143" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="1.5" stroke-dasharray="2 9"/>
    <circle class="spin2" cx="150" cy="150" r="97"  fill="none" stroke="rgba(255,255,255,.08)" stroke-width="1" stroke-dasharray="1 6"/>
    <g transform="rotate(135 150 150)">
      <circle cx="150" cy="150" r="115" fill="none" stroke="rgba(255,255,255,.07)" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="%%ARC%% %%CIRC%%"/>
      <circle id="arc" cx="150" cy="150" r="115" fill="none" stroke="url(#g)" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="0.01 %%CIRC%%" filter="url(#glow)"/>
    </g>
  </svg>
  <div class="center">
    <div class="cap">CHURN PROBABILITY</div>
    <div class="num"><span id="num">0.0</span><small>%</small></div>
    <div class="pill">%%LABEL%%</div>
  </div>
</div>
<script>
  const target = %%PCT%%, arcLen = %%ARC%% * %%PROB%%, circ = %%CIRC%%;
  const arc = document.getElementById('arc'), num = document.getElementById('num');
  setTimeout(() => { arc.style.strokeDasharray = arcLen + ' ' + circ; }, 200);
  let start = null;
  function ease(t){ return 1 - Math.pow(1 - t, 4); }
  function step(ts){
    if (!start) start = ts;
    const t = Math.min((ts - start) / 1800, 1);
    num.textContent = (target * ease(t)).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
  }
  setTimeout(() => requestAnimationFrame(step), 200);
</script>
"""
    for key, val in {
        "%%C1%%": c1, "%%C2%%": c2, "%%LABEL%%": label, "%%HOT%%": "hot" if prob >= 0.5 else "",
        "%%ARC%%": f"{arc:.2f}", "%%CIRC%%": f"{circ:.2f}", "%%PCT%%": f"{prob * 100:.1f}", "%%PROB%%": f"{prob:.4f}",
    }.items():
        html = html.replace(key, val)
    return html


def risk_signals(c: dict):
    """Plain-language flags tied to the model's top churn drivers."""
    flags = []
    if c["Contract"] == "Month-to-month":
        flags.append(("Month-to-month contract", True))
    else:
        flags.append((f"{c['Contract']} contract", False))
    if c["tenure"] < 12:
        flags.append(("Tenure under 12 months", True))
    elif c["tenure"] >= 36:
        flags.append(("Long-standing customer", False))
    if c["InternetService"] == "Fiber optic":
        flags.append(("Fiber optic plan", True))
    if c["InternetService"] != "No":
        if c["OnlineSecurity"] == "No":
            flags.append(("No online security", True))
        if c["TechSupport"] == "No":
            flags.append(("No tech support", True))
    if c["PaymentMethod"] == "Electronic check":
        flags.append(("Pays by electronic check", True))
    return flags


def retention_levers(c: dict, base: float):
    """What-if simulation: re-score the same customer after one realistic change."""
    results = []

    def trial(name, changes):
        p = predict_proba({**c, **changes})
        if p < base - 0.005:
            results.append((name, p, base - p))

    if c["Contract"] == "Month-to-month":
        trial("Move to a 1-year contract", {"Contract": "One year"})
        trial("Move to a 2-year contract", {"Contract": "Two year"})
    elif c["Contract"] == "One year":
        trial("Move to a 2-year contract", {"Contract": "Two year"})
    if c["InternetService"] != "No":
        changes = {k: "Yes" for k in ("OnlineSecurity", "TechSupport") if c[k] == "No"}
        if changes:
            trial("Add online security + tech support", changes)
    if c["PaymentMethod"] == "Electronic check":
        trial("Switch to automatic card payment", {"PaymentMethod": "Credit card (automatic)"})
    return sorted(results, key=lambda r: -r[2])


# =====================================================================
# 3. HERO
# =====================================================================
st.markdown("""
<div class="eyebrow">TELECOM · RETENTION INTELLIGENCE</div>
<h1 class="title">ChurnSense</h1>
<p class="subtitle">Score any customer's churn risk in real time, see what is driving it,
and test which retention move lowers it most.</p>
<div class="chips">
  <span class="chip"><b>Random Forest</b> model</span>
  <span class="chip">ROC-AUC <b>0.84</b></span>
  <span class="chip"><b>7,043</b> customers</span>
  <span class="chip"><b>5-fold</b> cross-validated</span>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_insights = st.tabs(["Predict", "Model Insights"])

# =====================================================================
# 4. PREDICT TAB
# =====================================================================
with tab_predict:
    left, right = st.columns(2, gap="large")
    cust = {}

    with left:
        with st.container(key="card_profile"):
            st.markdown('<div class="card-title"><span>01</span>Customer profile</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            cust["gender"] = c1.selectbox("Gender", ["Male", "Female"])
            cust["SeniorCitizen"] = int(c2.toggle("Senior citizen"))
            cust["Partner"] = yn(c1.toggle("Has partner"))
            cust["Dependents"] = yn(c2.toggle("Has dependents"))

        with st.container(key="card_contract"):
            st.markdown('<div class="card-title"><span>02</span>Contract & payment</div>', unsafe_allow_html=True)
            cust["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
            cust["Contract"] = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            cust["PaymentMethod"] = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            cust["PaperlessBilling"] = yn(st.toggle("Paperless billing", value=True))

    with right:
        with st.container(key="card_billing"):
            st.markdown('<div class="card-title"><span>03</span>Billing</div>', unsafe_allow_html=True)
            cust["MonthlyCharges"] = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
            cust["TotalCharges"] = round(cust["tenure"] * cust["MonthlyCharges"], 2)
            st.caption(f"Estimated lifetime charges: **${cust['TotalCharges']:,.0f}** (tenure × monthly)")

        with st.container(key="card_services"):
            st.markdown('<div class="card-title"><span>04</span>Services</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            cust["PhoneService"] = yn(s1.toggle("Phone service", value=True))
            if cust["PhoneService"] == "Yes":
                cust["MultipleLines"] = yn(s2.toggle("Multiple lines"))
            else:
                cust["MultipleLines"] = "No phone service"
            cust["InternetService"] = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

            addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            if cust["InternetService"] == "No":
                for a in addons:
                    cust[a] = "No internet service"
                st.caption("No internet service, so add-ons do not apply.")
            else:
                names = ["Online security", "Online backup", "Device protection", "Tech support", "Streaming TV", "Streaming movies"]
                a1, a2 = st.columns(2)
                for i, (a, n) in enumerate(zip(addons, names)):
                    cust[a] = yn((a1 if i % 2 == 0 else a2).toggle(n))

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    run = st.button("Analyse customer", width="stretch", type="primary")
    if run:
        st.session_state["ran"] = True

    # ------------------------- results -------------------------
    if st.session_state.get("ran"):
        prob = predict_proba(cust)
        _, label, c1_, c2_, headline, advice = tier_for(prob)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        g_col, v_col = st.columns([0.9, 1.1], gap="large")

        with g_col:
            with st.container(key="card_gauge"):
                render_html(gauge_html(prob), height=330)

        with v_col:
            flags_html = "".join(
                f'<span class="flag{"" if bad else " ok"}">{txt}</span>' for txt, bad in risk_signals(cust))
            st.markdown(f"""
            <div class="verdict" style="background:linear-gradient(135deg,{c1_}22,{c2_}11);
                 border:1px solid {c1_}66; box-shadow:0 0 36px {c1_}26;">
                <h3 style="color:{c1_};">{headline}</h3>
                <p>{advice}</p>
                <div class="flags">{flags_html}</div>
            </div>
            """, unsafe_allow_html=True)

            levers = retention_levers(cust, prob)
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="card-title"><span>↳</span>Retention levers: simulated impact</div>', unsafe_allow_html=True)
            if levers:
                for name, new_p, drop in levers:
                    st.markdown(f"""
                    <div class="lever"><div class="name">{name}</div>
                    <div class="shift">{prob:.0%} → {new_p:.0%} <small>(−{drop * 100:.0f} pts)</small></div></div>
                    """, unsafe_allow_html=True)
                st.markdown('<div class="note">Model-based what-if estimates from historical patterns, not guaranteed causal effects.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="note">No single change in the simulator lowers this customer\'s risk further.</div>',
                            unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#64748b;text-align:center;margin-top:1rem;">Set the customer details, then run the analysis.</p>',
                    unsafe_allow_html=True)

# =====================================================================
# 5. INSIGHTS TAB
# =====================================================================
with tab_insights:
    st.markdown('<div class="card-title"><span>↳</span>What drives churn in the model</div>', unsafe_allow_html=True)
    imp = pd.Series(model.feature_importances_, index=columns).sort_values().tail(10)
    fig = go.Figure(go.Bar(
        x=imp.values * 100, y=imp.index, orientation="h",
        marker=dict(color=imp.values, colorscale=[[0, "#4338ca"], [0.6, "#a855f7"], [1, "#2dd4bf"]]),
        text=[f"{v:.1f}%" for v in imp.values * 100], textposition="outside", textfont=dict(color="#cbd5e1"),
    ))
    fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=40), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", family="Inter, Arial, sans-serif"),
                      xaxis=dict(title="Importance (%)", gridcolor="rgba(255,255,255,.06)"),
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width="stretch")

    if os.path.exists("models/segment_summary.csv"):
        seg = pd.read_csv("models/segment_summary.csv")
        hot = seg["churn_rate"].idxmax()
        st.markdown('<div class="card-title" style="margin-top:1rem;"><span>↳</span>Customer segments (KMeans)</div>', unsafe_allow_html=True)
        cols = st.columns(len(seg), gap="medium")
        for i, row in seg.iterrows():
            cols[i].markdown(f"""
            <div class="seg {'hot' if i == hot else ''}">
              <div class="meta">SEGMENT {int(row['segment'])}{' · HIGHEST RISK' if i == hot else ''}</div>
              <div class="big">{row['churn_rate']:.0%} <span class="meta">churn</span></div>
              <div class="meta">{int(row['customers']):,} customers · {row['avg_tenure_months']:.0f} mo tenure ·
              ${row['avg_monthly_charges']:.0f}/mo</div>
            </div>""", unsafe_allow_html=True)

    if os.path.exists("models/cv_results.csv"):
        st.markdown('<div class="card-title" style="margin-top:1.4rem;"><span>↳</span>5-fold cross-validation</div>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv("models/cv_results.csv"), hide_index=True, width="stretch")

# =====================================================================
# 6. FOOTER
# =====================================================================
st.markdown("""
<div class="footer-wrapper">
  <div class="footer-credit">Built by
    <a href="https://www.linkedin.com/in/pratyushraj05" target="_blank">Pratyush Kant Raj</a>
    · <a href="https://github.com/PratyushRaj0512" target="_blank">GitHub</a>
  </div>
</div>
""", unsafe_allow_html=True)
