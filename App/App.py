"""
Student Performance Predictor - responsive, multilingual (Arabic / English / French)
Run:  streamlit run app.py      (after running train.py once)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from i18n import LANGS, RTL, T
from logic import build_row, get_tips, verdict

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",  # one comfortable column: works on phone, tablet and laptop
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- language
codes = list(LANGS.values())
default_code = st.query_params.get("lang", "ar")  # share links like ?lang=fr
default_idx = codes.index(default_code) if default_code in codes else 0
choice = st.radio(
    "language", list(LANGS), index=default_idx, horizontal=True, label_visibility="collapsed"
)
lang = LANGS[choice]
t = T[lang]
rtl = lang in RTL

# ---------------------------------------------------------------- styling
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp {{
    font-family: 'Cairo', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
.stApp {{ direction: {'rtl' if rtl else 'ltr'}; }}
[data-baseweb="slider"] {{ direction: ltr; }}            /* sliders always move left-to-right */
.block-container {{ max-width: 760px; padding: 1.2rem 1rem 3rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

.hero {{
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #fff; border-radius: 18px; padding: 1.4rem 1.2rem; margin: .4rem 0 1.2rem;
}}
.hero h1 {{ font-size: clamp(1.35rem, 5vw, 2rem); margin: 0 0 .3rem; color: #fff; padding: 0; }}
.hero p  {{ margin: 0; opacity: .92; font-size: clamp(.9rem, 3.4vw, 1.02rem); }}

.result {{
    border-radius: 16px; padding: 1.1rem 1.2rem; margin: 1rem 0 .6rem;
    border: 2px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent);
    text-align: center;
}}
.result .pct   {{ font-size: clamp(2.4rem, 12vw, 3.6rem); font-weight: 700; color: var(--c); line-height: 1.1; }}
.result .label {{ font-size: clamp(1rem, 4.2vw, 1.25rem); font-weight: 600; }}
.result .sub   {{ opacity: .75; font-size: .9rem; }}

/* bigger tap targets on touch screens */
div[data-testid="stFormSubmitButton"] button {{ min-height: 3rem; font-size: 1.05rem; font-weight: 600; border-radius: 12px; }}
.stSelectbox, .stNumberInput {{ margin-bottom: .2rem; }}
.disclaimer {{ opacity: .6; font-size: .8rem; text-align: center; margin-top: 1.5rem; }}

@media (max-width: 640px) {{
    .block-container {{ padding: .8rem .7rem 2.5rem; }}
    .hero {{ padding: 1.1rem .95rem; border-radius: 14px; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="hero"><h1>🎓 {t["title"]}</h1><p>{t["subtitle"]}</p></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- model
MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("outputs/metrics.json")

if not MODEL_PATH.exists():
    st.error(t["model_missing"])
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
yn = {"yes": t["yes"], "no": t["no"]}

# ---------------------------------------------------------------- form
with st.form("student_form"):
    st.subheader(t["main_inputs"])
    c1, c2 = st.columns(2)  # columns stack automatically on narrow screens
    with c1:
        studytime = st.selectbox(
            t["study_hours"], [1, 2, 3, 4], index=1, format_func=lambda v: t["study_opts"][v - 1]
        )
        failures = st.number_input(t["failures"], 0, 4, 0)
        absences = st.number_input(t["absences"], 0, 93, 4)
    with c2:
        higher = st.radio(t["higher"], ["yes", "no"], format_func=yn.get, horizontal=True)
        internet = st.radio(t["internet"], ["yes", "no"], format_func=yn.get, horizontal=True)
        schoolsup = st.radio(t["schoolsup"], ["no", "yes"], format_func=yn.get, horizontal=True)

    with st.expander(t["more_details"]):
        st.caption(t["scale"])
        d1, d2 = st.columns(2)
        with d1:
            age = st.number_input(t["age"], 15, 22, 17)
            sex = st.radio(t["sex"], ["F", "M"], format_func=lambda v: t[f"sex_{v}"], horizontal=True)
            address = st.radio(t["address"], ["U", "R"], format_func=lambda v: t[f"addr_{v}"], horizontal=True)
            traveltime = st.selectbox(t["traveltime"], [1, 2, 3, 4], format_func=lambda v: t["travel_opts"][v - 1])
            Medu = st.selectbox(t["Medu"], [0, 1, 2, 3, 4], index=2, format_func=lambda v: t["edu_opts"][v])
            Fedu = st.selectbox(t["Fedu"], [0, 1, 2, 3, 4], index=2, format_func=lambda v: t["edu_opts"][v])
            famsup = st.radio(t["famsup"], ["yes", "no"], format_func=yn.get, horizontal=True)
            paid = st.radio(t["paid"], ["no", "yes"], format_func=yn.get, horizontal=True)
        with d2:
            health = st.slider(t["health"], 1, 5, 3)
            famrel = st.slider(t["famrel"], 1, 5, 4)
            freetime = st.slider(t["freetime"], 1, 5, 3)
            goout = st.slider(t["goout"], 1, 5, 3)
            Dalc = st.slider(t["Dalc"], 1, 5, 1)
            Walc = st.slider(t["Walc"], 1, 5, 1)
            activities = st.radio(t["activities"], ["no", "yes"], format_func=yn.get, horizontal=True)
            romantic = st.radio(t["romantic"], ["no", "yes"], format_func=yn.get, horizontal=True)

    submitted = st.form_submit_button(f"🔮 {t['predict']}", type="primary", use_container_width=True)

# ---------------------------------------------------------------- result
if submitted:
    values = dict(
        studytime=studytime, failures=failures, absences=absences, higher=higher,
        internet=internet, schoolsup=schoolsup, age=age, sex=sex, address=address,
        traveltime=traveltime, Medu=Medu, Fedu=Fedu, famsup=famsup, paid=paid,
        health=health, famrel=famrel, freetime=freetime, goout=goout, Dalc=Dalc,
        Walc=Walc, activities=activities, romantic=romantic,
    )
    p_pass = float(model.predict_proba(build_row(values, model))[0, 1])
    level = verdict(p_pass)
    color = {"pass": "#16a34a", "border": "#d97706", "fail": "#dc2626"}[level]
    icon = {"pass": "🎉", "border": "⚖️", "fail": "⚠️"}[level]

    st.markdown(
        f"""
<div class="result" style="--c:{color}">
  <div class="sub">{t['prob_pass']}</div>
  <div class="pct">{p_pass:.0%}</div>
  <div class="label">{icon} {t['res_' + level]}</div>
</div>""",
        unsafe_allow_html=True,
    )
    st.progress(p_pass)

    st.markdown(f"**💡 {t['tips_title']}**")
    for key in get_tips(values):
        st.markdown(f"- {t[key]}")

# ---------------------------------------------------------------- model info
if METRICS_PATH.exists():
    with st.expander(f"📊 {t['perf_title']}"):
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"**{t['best_model']}:** {data['best_model']}")
        st.dataframe(pd.DataFrame(data["results"]).T, use_container_width=True)
        for img in ["confusion_matrix.png", "feature_importance.png"]:
            p = Path("outputs") / img
            if p.exists():
                st.image(str(p), use_container_width=True)

st.markdown(f'<div class="disclaimer">{t["disclaimer"]}</div>', unsafe_allow_html=True)