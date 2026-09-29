# =====================================================================
#  HeartFailure Clinical Explorer
#  Team Python Pioneers | NumpyNinja Python Hackathon
#
#  Run:  streamlit run DashboardHeartfailure.py
#  Data: Cardiac_Cleaned_Data.xlsb in the same folder
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import html
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

st.set_page_config(page_title="HeartFailure Clinical Explorer", page_icon="❤️", layout="wide")

# ----------------------------- COLOURS (from our original file) -----------------------------
NAVY = "#073B4C"       # dark teal / headings
TEAL = "#0B5D6B"
GREEN = "#087F5B"
TEAL2 = "#0B7A75"
BLUE = "#087F9B"
GREYTXT = "#637B83"
BG = "#F4F9FB"
ALERT = "#D1495B"      # only for danger / death highlights
RAMP = ["#B7E4D8", "#6CC3B0", TEAL2, TEAL, NAVY]      # light = better, dark = worse
READMIT, DEATH = BLUE, ALERT                            # same meaning on every chart

# ----------------------------- STYLE -----------------------------
st.markdown(f"""
<style>
.stApp {{background:{BG};}}
section[data-testid="stSidebar"] {{background:linear-gradient(180deg,#073B4C,#0B5D6B,#087F5B);}}
section[data-testid="stSidebar"] * {{color:white !important;}}

/* Sidebar navigation styling */
section[data-testid="stSidebar"] div[data-testid="stRadio"], section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{width:100%;}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{gap:14px; width:100%; display:flex; flex-direction:column; align-items:stretch;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.35);
    border-radius:14px; padding:16px 18px; width:100% !important; max-width:100% !important; display:flex !important; box-sizing:border-box; justify-content:center; text-align:center; transition:0.2s;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{background:rgba(255,255,255,0.18);}}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {{
    background:rgba(255,255,255,0.25); border:1px solid white;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {{display:none;}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div {{margin:0 auto;}}
section[data-testid="stSidebar"] div[role="radiogroup"] p {{font-size:16px; font-weight:600;}}

.hdr {{background:linear-gradient(90deg,#073B4C,#087F5B);color:white;padding:25px 30px;border-radius:16px;margin-bottom:20px;}}
.hdr h1 {{margin:0;font-size:30px;color:white}} .hdr p {{margin:6px 0 0;opacity:.92}}
.section {{background:white;padding:20px;border-radius:15px;box-shadow:0 3px 12px rgba(0,0,0,.06);margin-bottom:18px;}}
.kpi {{background:white;padding:16px;border-radius:14px;border-left:5px solid #087F5B;box-shadow:0 3px 12px rgba(0,0,0,.06);min-height:105px;}}
.kpi .i{{font-size:25px}} .kpi .t{{font-size:13px;color:#637B83;font-weight:600}} .kpi .v{{font-size:26px;color:#073B4C;font-weight:700}}
.found {{background:#EAF5F8;border-left:5px solid #087F9B;padding:14px 16px;border-radius:9px;margin:6px 0;}}
.todo {{background:#E8F6EF;border-left:5px solid #087F5B;padding:14px 16px;border-radius:9px;margin:6px 0;}}
.badge {{display:inline-block;background:#073B4C;color:white;padding:3px 10px;border-radius:20px;font-size:12px;margin-bottom:6px;}}
.member {{background:white;border-radius:14px;padding:18px;text-align:center;box-shadow:0 3px 12px rgba(0,0,0,.06);border-top:5px solid #087F5B;}}
.member .n {{font-size:17px;font-weight:700;color:#073B4C}} .member .r {{font-size:13px;color:#637B83}}
.stTabs [data-baseweb="tab"] p {{font-size:15px;}}
/* ---------- HeartFailure Clinical Explorer styling ---------- */
.hero {{background:linear-gradient(120deg,#FFFFFF 0%,#EAF5F8 55%,#D6EFE6 100%);border-radius:22px;padding:58px 54px 0 54px;
        box-shadow:0 6px 20px rgba(7,59,76,.10);overflow:hidden;position:relative;}}
.hero .t1 {{font-size:72px;font-weight:900;color:#073B4C;line-height:1;letter-spacing:1px;margin:0;}}
.hero .t2 {{font-size:58px;font-weight:900;color:#087F5B;line-height:1.1;margin:6px 0 0 0;}}
.hero .sub {{font-size:20px;color:#0B5D6B;margin-top:14px;}}
.hero .line {{height:4px;width:70%;background:linear-gradient(90deg,#073B4C,#087F5B);border-radius:4px;margin:18px 0 26px 0;}}
.pill {{display:inline-block;background:#073B4C;color:white;font-size:30px;font-weight:800;padding:10px 30px;border-radius:14px;letter-spacing:1px;}}
.meet {{color:#087F5B;font-weight:800;font-size:21px;letter-spacing:1px;margin:10px 0 18px 0;}}
.tm {{display:flex;align-items:center;gap:14px;padding:6px 4px;}}
.tm .av {{width:76px;height:76px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:30px;color:white;flex-shrink:0;}}
.tm .nm {{font-size:21px;font-weight:800;}} .tm .rl {{font-size:16px;color:#637B83;border-top:3px solid;padding-top:4px;margin-top:4px;}}
.herobar {{background:#073B4C;color:white;border-radius:18px 18px 0 0;display:flex;justify-content:space-around;padding:24px 10px;margin:34px -54px 0 -54px;font-size:21px;font-weight:600;}}
.bigtitle {{text-align:center;font-size:38px;font-weight:900;color:#073B4C;letter-spacing:1px;margin:0;}}
.bigtitle span {{display:inline-block;width:18%;height:3px;background:#073B4C;vertical-align:middle;margin:0 18px;border-radius:3px;}}
.lead {{max-width:1050px;margin:6px auto 12px auto;text-align:center;font-size:14px;color:#0B5D6B;font-weight:500;}}
.spec {{background:#073B4C;color:white;border-radius:18px;padding:18px 18px 8px 18px;}}
.spec h3 {{color:#7FD8BE;margin:0 0 10px 0;font-size:22px;}}
.spec .row {{display:flex;gap:12px;align-items:center;border-top:1px solid rgba(255,255,255,.18);padding:10px 0;}}
.spec .ic {{font-size:24px;width:34px;text-align:center;}} .spec .k {{font-weight:700;}} .spec .v {{opacity:.9;font-size:14px;}}
.card-h {{text-align:center;}} .card-h .ic {{font-size:34px;}} .card-h .nm {{font-weight:900;font-size:15px;letter-spacing:.5px;margin:4px 0 6px 0;}}
.card-h ul {{text-align:left;font-size:13px;color:#073B4C;padding-left:18px;margin:0;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{background:white;border-radius:16px !important;}}
.checkbox {{background:#EAF5F8;border-left:6px solid #073B4C;border-radius:16px;padding:22px 26px;box-shadow:0 3px 12px rgba(0,0,0,.05);margin-bottom:18px;}}
.checkbox b.h {{font-size:18px;color:#073B4C;}}
.checkbox .it {{font-size:17px;color:#073B4C;margin:16px 0;}}
.pagetitle {{font-size:44px;font-weight:800;color:#073B4C;margin:10px 0 18px 0;}}
.dash-title {{font-size:52px;font-weight:800;color:#073B4C;margin:0 0 18px 0;}}
.kpi2 {{background:white;border-radius:16px;padding:14px 18px;box-shadow:0 4px 16px rgba(7,59,76,.08);min-height:92px;}}
.kpi2 .t {{font-size:14px;color:#1F2D33;}} .kpi2 .v {{font-size:28px;color:#073B4C;margin-top:6px;}}
.sec {{font-size:32px;font-weight:700;color:#073B4C;margin:14px 0 6px 0;}}
.sec .badge {{font-size:13px;vertical-align:middle;margin-left:8px;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] div,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{color:#073B4C !important; fill:#073B4C !important;}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{background:white;border-radius:10px;}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] button,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {{color:#073B4C !important; -webkit-text-fill-color:#073B4C !important; fill:#073B4C !important;}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {{background:white !important;border-radius:10px;}}

/* Presentation sizing */
.hero {{min-height:520px;padding-top:42px;}}
.hero .t1 {{font-size:62px;}}
.hero .t2 {{font-size:46px;}}
.hero .sub {{font-size:18px;}}
.tm .av {{width:62px;height:62px;font-size:28px;}}
.tm .nm {{font-size:18px;}}
.tm .rl {{font-size:14px;}}
.herobar {{padding:18px 10px;font-size:18px;}}
.overview-fit {{margin-top:-8px;}}
.qbox {{background:white;border:2px solid #087F5B;border-radius:12px;padding:12px 16px;margin:6px 0 12px 0;font-size:16px;color:#073B4C;}}
.stSelectbox label {{font-size:17px !important;font-weight:800 !important;color:#073B4C !important;}}

</style>
""", unsafe_allow_html=True)


# ----------------------------- TEAM LOGO -----------------------------
LOGO_PATH = Path(__file__).parent / "numpy_ninja_logo.png"


# ----------------------------- SMALL HELPERS -----------------------------
def kpi(icon, title, value, size=26):
    st.markdown(f"<div class='kpi'><div class='i'>{icon}</div><div class='t'>{title}</div>"
                f"<div class='v' style='font-size:{size}px'>{value}</div></div>", unsafe_allow_html=True)


def found(text):
    st.markdown(f"<div class='found'><b>What we found:</b> {text}</div>", unsafe_allow_html=True)


def todo(text):
    st.markdown(f"<div class='todo'><b>Action:</b> {text}</div>", unsafe_allow_html=True)


def kpi2(icon, title, value):
    """Plain white KPI card for the HeartFailure dashboard."""
    st.markdown(f"<div class='kpi2'><div class='t'>{icon} {title}</div><div class='v'>{value}</div></div>",
                unsafe_allow_html=True)


def section(title, kind):
    """Tab heading with a small Descriptive / Prescriptive / Predictive label."""
    st.markdown(f"<div class='sec'>{title} <span class='badge'>{kind}</span></div>", unsafe_allow_html=True)


def badge(text):
    st.markdown(f"<span class='badge'>{text}</span>", unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, title_font_color=NAVY,
                      font_color=NAVY, margin=dict(t=60, l=10, r=10, b=10), legend_title="")
    return fig


def bar(x, y, title, colours, ytitle="% of patients", fmt=".1f", height=380):
    fig = px.bar(x=x, y=y, text_auto=fmt, color=x, color_discrete_sequence=colours, title=title)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=ytitle)
    return style(fig, height)


def two_outcomes(table, title):
    long = table.reset_index().melt(id_vars=table.index.name, var_name="Outcome", value_name="Percent")
    fig = px.bar(long, x=table.index.name, y="Percent", color="Outcome", barmode="group", text_auto=".1f",
                 color_discrete_map={"Readmitted in 6 months": READMIT, "Died in 6 months": DEATH, "Came back": READMIT, "Died": DEATH}, title=title)
    fig.update_layout(yaxis_title="% of patients", xaxis_title="")
    return style(fig, 400)


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------- DATA -----------------------------
HERE = Path(__file__).parent


@st.cache_data
def load_data():
    data_file = HERE / "Cardiac_Cleaned_Data.xlsb"
    if not data_file.exists():
        raise FileNotFoundError("Cardiac_Cleaned_Data.xlsb was not found beside the dashboard file.")
    df = pd.read_excel(data_file, engine="pyxlsb")
    new = {}

    stage_order = ["G1 (>=90)", "G2 (60-89)", "G3a (45-59)", "G3b (30-44)", "G4 (15-29)", "G5 (<15)"]
    ckd = pd.cut(df["glomerular_filtration_rate"], bins=[0, 15, 30, 45, 60, 90, 1000], right=False, labels=stage_order[::-1])
    new["ckd_stage"] = pd.Categorical(ckd, categories=stage_order, ordered=True)

    cut = np.where(df["gender"] == "Male", 130, 120)
    hb = df["hemoglobin"]
    anemia = np.select([hb.isna(), hb >= cut, hb >= 110, hb >= 80], ["Missing", "No anemia", "Mild", "Moderate"], default="Severe")
    new["anemia_level"] = pd.Categorical(pd.Series(anemia).replace("Missing", np.nan),
                                         categories=["No anemia", "Mild", "Moderate", "Severe"], ordered=True)

    sbp, dbp = df["systolic_blood_pressure"], df["diastolic_blood_pressure"]
    bp_order = ["Low (<90)", "Normal", "Elevated", "High stage 1", "High stage 2"]
    bp = np.select([sbp.isna(), sbp < 90, (sbp >= 140) | (dbp >= 90), (sbp >= 130) | (dbp >= 80), sbp >= 120],
                   ["Missing", bp_order[0], bp_order[4], bp_order[3], bp_order[2]], default=bp_order[1])
    new["bp_stage"] = pd.Categorical(pd.Series(bp).replace("Missing", np.nan), categories=bp_order, ordered=True)

    new["age"] = df["agecat"].apply(lambda s: (int(str(s).split("-")[0]) + int(str(s).split("-")[1])) / 2)
    new["male"] = (df["gender"] == "Male").astype(int)
    new["nlr"] = df["neutrophil_count"] / df["lymphocyte_count"]
    new["nlr_log"] = np.log(new["nlr"])
    new["troponin_log"] = np.log1p(df["high_sensitivity_troponin"])
    if "bnp_log" not in df.columns:
        new["bnp_log"] = np.log1p(df["brain_natriuretic_peptide"])
    return pd.concat([df, pd.DataFrame(new)], axis=1)


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the cleaned data file: {e}")
    st.stop()

# Model inputs (admission-time data only)
DEATH_FEATURES = ["nyha_cardiac_function_classification", "killip_grade", "bnp_log", "troponin_log",
                  "nlr_log", "albumin", "hemoglobin", "sodium"]
READMIT_FEATURES = ["nyha_cardiac_function_classification", "killip_grade", "systolic_blood_pressure", "pulse",
                    "respiration", "glomerular_filtration_rate", "urea", "cystatin",
                    "moderate_to_severe_chronic_kidney_disease", "bnp_log", "troponin_log", "nlr_log", "albumin",
                    "hemoglobin", "sodium", "cci_score", "diabetes", "chronic_obstructive_pulmonary_disease",
                    "age", "male", "bmi"]


def logistic():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))])


def model_set():
    return {
        "Logistic Regression": logistic(),
        "Random Forest": Pipeline([("impute", SimpleImputer(strategy="median")),
                                   ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                                                    class_weight="balanced_subsample", random_state=0, n_jobs=-1))]),
        "ANN (neural network)": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                          ("model", MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=2000, random_state=0))]),
    }


@st.cache_data
def cv_probs(data, features, target, model_name, repeats=1):
    """Risk for every patient, predicted by a model that never saw that patient (5-fold cross-validation)."""
    X, y = data[features].astype(float), data[target]
    probs = []
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs.append(cross_val_predict(model_set()[model_name], X, y, cv=cv, method="predict_proba")[:, 1])
    return np.mean(probs, axis=0)


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.image(LOGO_PATH, width=110)
    st.markdown("<div style='text-align:center;font-size:48px'>❤️</div>"
                "<h2 style='text-align:center;margin:0'>HeartFailure</h2>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["🏠 Introduction", "📘 Data Overview", "🧹 Data Cleaning & Feature Engineering",
                                   "🩺 Interactive Clinical Insights", "🤖 Model Performance", "📌 Key Takeaways & Conclusion"],
                    label_visibility="collapsed")


# =====================================================================
# 1. INTRODUCTION
# =====================================================================
if page == "🏠 Introduction":
    team = [("Aditi Mishra", "Team Lead", NAVY), ("Saranya Shanmugam", "Team Member", GREEN),
            ("Sashi Laguduva", "Team Member", BLUE), ("Sudha Madhuri Basa", "Team Member", ALERT)]
    members = "".join(
        f"<div class='tm'><div class='av' style='background:{c}'>👤</div>"
        f"<div><div class='nm' style='color:{c}'>{n}</div><div class='rl' style='border-color:{c}'>{r}</div></div></div>"
        for n, r, c in team)
    heart_svg = (
        "<svg viewBox='0 0 220 200' width='300' style='position:absolute;right:50px;top:40px;opacity:.95'>"
        "<defs><linearGradient id='hg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#E86A7A'/>"
        "<stop offset='1' stop-color='#B8324A'/></linearGradient></defs>"
        "<path d='M110 185 C 30 125, 5 75, 40 38 C 70 8, 102 22, 110 50 C 118 22, 150 8, 180 38 C 215 75, 190 125, 110 185 Z' fill='url(#hg)'/>"
        "<polyline points='20,105 70,105 85,80 100,135 118,55 135,120 148,105 200,105' fill='none' stroke='white' "
        "stroke-width='7' stroke-linejoin='round' stroke-linecap='round'/></svg>")
    st.markdown(
        f"<div class='hero'>{heart_svg}"
        "<p class='t1'>HEART FAILURE</p>"
        "<p class='t2'>CLINICAL OUTCOME ANALYTICS</p>"
        "<div class='sub'>Spotting high-risk heart failure patients on the day they are admitted</div>"
        "<div class='line'></div>"
        "<div style='text-align:center'><span class='pill'>TEAM 2: PYTHONPIONEERS</span>"
        "<div class='meet'>—— MEET OUR TEAM ——</div></div>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px'>{members}</div>"
        "<div class='herobar'><span>⭐ Early Risk Detection</span><span>❤️ Better Decisions</span>"
        "<span>👥 Healthier Hearts</span></div></div>", unsafe_allow_html=True)


# =====================================================================
# 2. DATA OVERVIEW
# =====================================================================
elif page == "📘 Data Overview":
    st.markdown("<div class='bigtitle'><span></span>DATA OVERVIEW<span></span></div>"
                "<div class='lead'>This dataset links 7 hospital tables for 2,008 heart failure patients: who they are, "
                "how sick their heart is, their other diseases, 100+ blood tests, alertness, medicines, and what happened "
                "to them up to 6 months after discharge. It lets us find who needs extra care and spot them early.</div>",
                unsafe_allow_html=True)

    years = pd.to_datetime(df["admission_date"])
    spec_rows = [("👥", "Patients", f"{len(df):,} hospitalised heart failure patients"),
                 ("🗂️", "Source", "7 hospital tables, linked by patient ID"),
                 ("📅", "Admissions", f"{years.dt.year.min()} – {years.dt.year.max()}"),
                 ("⏱️", "Follow-up", "28 days, 3 months, 6 months"),
                 ("🧪", "Tests", "100+ blood tests and vital signs"),
                 ("💊", "Medicines", "25 drugs given in hospital"),
                 ("📋", "Final table", "2,008 rows × 210 columns")]
    spec = "".join(f"<div class='row'><div class='ic'>{i}</div><div><div class='k'>{k}:</div><div class='v'>{v}</div></div></div>"
                   for i, k, v in spec_rows)

    left, right = st.columns([1, 3.2])
    with left:
        st.markdown(f"<div class='spec'><h3>Cardiac Failure<br>Dataset Specifications</h3>{spec}</div>", unsafe_allow_html=True)

    def mini(fig):
        fig.update_layout(template="plotly_white", height=150, margin=dict(t=5, l=5, r=5, b=5), showlegend=False,
                          xaxis_title="", yaxis_title="", font_size=10)
        fig.update_traces(selector=dict(type="pie"), textinfo="none")
        return fig

    cards = [
        ("🧍", "DEMOGRAPHY", NAVY, ["Gender", "Age group", "Height, weight, BMI", "Occupation"],
         lambda: px.bar(df["agecat"].value_counts().sort_index(), color_discrete_sequence=[NAVY])),
        ("❤️", "CARDIAC", ALERT, ["NYHA class (symptoms)", "Killip grade (fluid/shock)", "Heart failure type", "Heart scan (LVEF)"],
         lambda: px.bar(df["nyha_cardiac_function_classification"].value_counts().sort_index(), color_discrete_sequence=[ALERT])),
        ("📜", "HISTORY", GREEN, ["Diabetes", "Kidney disease", "COPD, liver disease", "Comorbidity score"],
         lambda: px.bar(pd.Series({"Kidney": df["moderate_to_severe_chronic_kidney_disease"].mean(),
                                   "Diabetes": df["diabetes"].mean(),
                                   "COPD": df["chronic_obstructive_pulmonary_disease"].mean()}) * 100,
                        color_discrete_sequence=[GREEN])),
        ("🏥", "HOSPITAL STAY", BLUE, ["Admission type", "Days in hospital", "Death: 28d / 3m / 6m", "Readmission: 28d / 3m / 6m"],
         lambda: px.bar(pd.Series({"Came back": df["re_admission_within_6_months"].mean(),
                                   "Died": df["death_within_6_months"].mean()}) * 100,
                        color=["Came back", "Died"], color_discrete_sequence=[READMIT, DEATH])),
        ("🧪", "LABS", TEAL2, ["BNP (heart strain)", "Troponin (heart damage)", "Kidney tests (eGFR)", "Blood count, salts"],
         lambda: px.histogram(np.log10(df["brain_natriuretic_peptide"].dropna()), nbins=25, color_discrete_sequence=[TEAL2])),
        ("🧠", "RESPONSIVENESS", "#6C4AB6", ["Eye opening", "Verbal response", "Movement", "GCS score (alertness)"],
         lambda: px.pie(values=df["gcs_category"].value_counts().values, names=df["gcs_category"].value_counts().index,
                        hole=.6, color_discrete_sequence=["#6C4AB6", "#B9A6E3", "#D8CCF1", "#EDE7F8"])),
        ("💊", "PRESCRIPTIONS", "#E07A5F", ["25 medicines", "Water tablets", "Heart medicines", "Medicines per patient"],
         lambda: px.histogram(df["total_drugs"], nbins=16, color_discrete_sequence=["#E07A5F"])),
        ("✨", "DERIVED FEATURES", TEAL, ["BMI / BP groups", "Kidney stage, anemia level", "Warning flags", "NLR, comorbidity count"],
         lambda: px.pie(values=df["bmi_category"].value_counts().values, names=df["bmi_category"].value_counts().index,
                        hole=.6, color_discrete_sequence=[TEAL, "#6CC3B0", "#B7E4D8", NAVY])),
    ]
    with right:
        for row in (cards[:4], cards[4:]):
            cols = st.columns(4)
            for col, (ic, nm, colr, items, chart) in zip(cols, row):
                with col:
                    with st.container(border=True):
                        bullets = "".join(f"<li>{x}</li>" for x in items)
                        st.markdown(f"<div class='card-h'><div class='ic'>{ic}</div>"
                                    f"<div class='nm' style='color:{colr}'>{nm}</div><ul>{bullets}</ul></div>",
                                    unsafe_allow_html=True)
                        st.plotly_chart(mini(chart()), width="stretch", config={"displayModeBar": False})


# =====================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =====================================================================
elif page == "🧹 Data Cleaning & Feature Engineering":
    st.markdown("<div class='pagetitle'>🧹 Data Cleaning & Feature Engineering</div>", unsafe_allow_html=True)
    steps = ["Removed a fake patient record and joined all 7 tables into one (one row per patient)",
             "Set impossible values to blank: 0 kg weight, 0 pulse, BMI of 404, reversed blood pressure",
             "Fixed wrong units: troponin, hematocrit and heart-scan values",
             "Filled blanks only when the meaning was clear (blank breathing support = no ventilation)",
             "Kept real gaps empty: missing lab tests were not invented",
             "Changed medicines from many rows per patient to one row per patient",
             "Renamed confusing lab columns and made yes/no columns 1/0"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in steps)
    st.markdown(f"<div class='checkbox'><b class='h'>Data Cleaning Steps:</b>{items}</div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:#073B4C'>🧠 Engineered Features</h3>", unsafe_allow_html=True)
    feats = pd.DataFrame({
        "Feature": ["bmi_category, bp_category", "ckd_stage, anemia_level", "bnp_elevated_flag, troponin_elevated_flag",
                    "polypharmacy_flag, total_drugs", "comorbidity_count", "nlr (neutrophil ÷ lymphocyte)", "bnp_log, hs_crp_log"],
        "Purpose": ["Compare patient groups easily", "Kidney and blood health in clear stages",
                    "Quick yes/no warning signs (heart strain, heart damage)", "How many medicines each patient takes",
                    "How much extra illness a patient carries", "Free inflammation marker from the routine blood count",
                    "Stop a few extreme values from controlling the models"]})
    st.dataframe(feats, hide_index=True, width="stretch")


# =====================================================================
# 4. INTERACTIVE CLINICAL INSIGHTS  (guided: Insight Area -> Marker -> Outcome)
# =====================================================================
elif page == "🩺 Interactive Clinical Insights":
    st.markdown("<div class='dash-title'>🩺 Interactive Clinical Insights</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi2("🔁", "Came back (6 months)", pct(df["re_admission_within_6_months"].mean()))
    with c2: kpi2("⚠️", "Died (6 months)", pct(df["death_within_6_months"].mean()))
    with c3: kpi2("❤️", "Severe symptoms (NYHA 3–4)", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))
    with c4: kpi2("🧪", "Median BNP", f"{df['brain_natriuretic_peptide'].median():.0f}")
    st.write("")

    # ---------------- helper to cut a column into labelled groups ----------------
    def cut(col, bins, labels):
        return pd.cut(df[col], bins=bins, labels=labels, right=False)

    def yes_no(mask, yes, no):
        return pd.Series(np.where(mask, yes, no), index=df.index)

    def drug_any(cols):
        return df[cols].sum(axis=1) > 0

    nyha_grp = np.where(df["nyha_cardiac_function_classification"] == 4, "Symptoms at rest", "Symptoms on activity")
    kil_grp = np.where(df["killip_grade"] >= 3, "fluid/shock", "no fluid")

    # ---------------- all insight areas, markers, groups and plain-English meaning ----------------
    # Each marker: (kind, function returning groups, "what it means" text)
    AREAS = {
        "❤️ Cardiac Biomarkers": ("Prescriptive", {
            "BNP (heart strain)": (lambda: cut("brain_natriuretic_peptide", [0, 100, 500, 2000, 1e9],
                                               ["Normal (<100)", "100–500", "500–2000", "Very high (≥2000)"]),
                                   "BNP rises when the heart is stretched. Higher BNP means a more strained heart; "
                                   "use it with the bedside exam to judge how sick the patient is."),
            "Troponin (heart damage)": (lambda: cut("high_sensitivity_troponin", [0, 14.0001, 100, 1e9],
                                                    ["Normal (≤14)", "Raised (14–100)", "Very high (>100)"]),
                                        "Troponin shows heart muscle damage. Very high troponin points to acute injury "
                                        "and a patient who needs closer monitoring."),
        }),
        "🫘 Kidney Function": ("Prescriptive", {
            "Kidney stage (eGFR)": (lambda: df["ckd_stage"],
                                    "Heart and kidneys pull each other down. eGFR below 45 (stage G3b or worse) should be "
                                    "treated as high risk: careful water-tablet dosing, potassium checks, early follow-up."),
            "Urea": (lambda: cut("urea", [0, 7.1, 15, 1e9], ["Normal (<7.1)", "Raised (7.1–15)", "High (≥15)"]),
                     "Urea builds up when the kidneys are not clearing waste, often because the heart pumps poorly."),
            "Chronic kidney disease (history)": (lambda: yes_no(df["moderate_to_severe_chronic_kidney_disease"] == 1, "Yes", "No"),
                                                 "Known kidney disease adds long-term burden and limits which heart "
                                                 "medicines can be used safely."),
        }),
        "🔥 Inflammation & Nutrition": ("Prescriptive", {
            "NLR (routine blood count)": (lambda: pd.qcut(df["nlr"], 4, labels=["Lowest", "Low", "High", "Highest (≥8.7)"]),
                                          "NLR is free from the routine blood count and available for almost every patient. "
                                          "Flag NLR of 8.7 or more for closer monitoring."),
            "White blood cells": (lambda: cut("white_blood_cell", [0, 4, 10, 1e9], ["Low (<4)", "Normal (4–10)", "High (>10)"]),
                                  "A high white cell count suggests infection or stress, a common trigger of heart failure attacks."),
            "hs-CRP (special test)": (lambda: cut("hs_crp", [0, 3, 1e9], ["Normal (<3)", "High (≥3)"]),
                                      "hs-CRP measures inflammation but was not tested for about half of patients, "
                                      "so NLR is the more practical marker."),
            "Albumin (nutrition)": (lambda: cut("albumin", [0, 35, 1e9], ["Low (<35)", "Normal (≥35)"]),
                                    "Low albumin reflects poor nutrition and inflammation; these patients recover less well."),
        }),
        "🛏️ Clinical Severity (Killip / NYHA)": ("Predictive", {
            "Killip grade (fluid / shock)": (lambda: df["killip_grade"].map(lambda k: f"Killip {k}"),
                                             "A 30-second bedside exam. Killip 1 patients are low risk; Killip 3–4 "
                                             "(fluid in lungs or shock) need close monitoring."),
            "NYHA class (symptoms)": (lambda: df["nyha_cardiac_function_classification"].map(lambda k: f"NYHA {k}"),
                                      "NYHA shows how much symptoms limit daily life. Class 4 (symptoms at rest) carries the most risk."),
            "Killip + NYHA together": (lambda: pd.Series(pd.Categorical([f"{n} + {k}" for n, k in zip(nyha_grp, kil_grp)], categories=[
                                           "Symptoms on activity + no fluid", "Symptoms at rest + no fluid",
                                           "Symptoms on activity + fluid/shock", "Symptoms at rest + fluid/shock"],
                                           ordered=True), index=df.index),
                                       "Using both bedside scores together separates patients even better than either alone."),
            "Alertness (consciousness)": (lambda: yes_no(df["consciousness"] == "Clear", "Fully alert", "Not fully alert"),
                                          "Patients who are not fully alert at admission are rare but very high risk."),
        }),
        "🕰️ Current Severity vs Prior History": ("Predictive", {
            "Old heart attack": (lambda: yes_no(df["myocardial_infarction"] == 1, "Yes", "No"),
                                 "Past diagnoses tell us little about who will die. Today's bedside condition matters more."),
            "Past heart failure": (lambda: yes_no(df["congestive_heart_failure"] == 1, "Yes", "No"),
                                   "Most patients (93%) already had heart failure before. Patients newly diagnosed at this admission "
                                   "had more deaths, so a first-time diagnosis deserves extra attention."),
            "Circulation problems (PVD)": (lambda: yes_no(df["peripheral_vascular_disease"] == 1, "Yes", "No"),
                                           "Old vascular disease adds little once current severity is known."),
            "Killip grade today (compare)": (lambda: df["killip_grade"].map(lambda k: f"Killip {k}"),
                                             "Compare with the history markers: today's Killip grade shows a much bigger difference."),
        }),
        "🩸 Anemia": ("Prescriptive", {
            "Anemia level (WHO)": (lambda: df["anemia_level"],
                                   "Mild and moderate anemia add little risk, but severe anemia (Hb below 80) is a real "
                                   "warning sign: flag it at admission and correct it."),
            "Severe anemia vs rest": (lambda: pd.Series(np.where(df["anemia_level"].isna(), None,
                                                                 np.where(df["anemia_level"] == "Severe", "Severe (<80)", "Not severe")),
                                                        index=df.index),
                                      "Severe anemia makes a weak heart work much harder to deliver oxygen."),
        }),
        "🩺 Blood Pressure": ("Prescriptive", {
            "Blood pressure stage": (lambda: df["bp_stage"],
                                     "Low BP (below 90) means the pump is failing: treat as possible shock. Higher BP patients "
                                     "come back less often because their heart still has strength."),
            "Pulse": (lambda: cut("pulse", [0, 60, 100.0001, 1e9], ["Slow (<60)", "Normal (60–100)", "Fast (>100)"]),
                      "A fast pulse can mean the heart is struggling to keep up."),
        }),
        "🧂 Blood Gas & Salts": ("Prescriptive", {
            "Sodium": (lambda: cut("sodium", [0, 135, 145.0001, 1e9], ["Low (<135)", "Normal (135–145)", "High (>145)"]),
                       "Low sodium often reflects fluid overload; review fluids and water tablets."),
            "Potassium": (lambda: cut("potassium", [0, 3.5, 5.0001, 1e9], ["Low (<3.5)", "Normal (3.5–5)", "High (>5)"]),
                          "High potassium is common with weak kidneys and some heart medicines; monitor it closely."),
            "Lactate (blood gas)": (lambda: cut("lactate", [0, 2, 1e9], ["Normal (<2)", "High (≥2)"]),
                                    "High lactate means tissues are short of oxygen. Tested for about half of patients."),
            "Bicarbonate (blood gas)": (lambda: cut("standard_bicarbonate", [0, 22, 1e9], ["Low (<22)", "Normal (≥22)"]),
                                        "Low bicarbonate means acid build-up in the blood, a sign of a very sick patient."),
        }),
        "👥 Patient Profile": ("Descriptive", {
            "Age group": (lambda: df["agecat"], "Heart failure risk in this group follows how sick patients are more than their age."),
            "Gender": (lambda: df["gender"], "Women make up 58% of patients; outcomes differ little by gender."),
            "BMI group": (lambda: df["bmi_category"].astype(object),
                          "1 in 4 patients is underweight, a sign of frailty in long-term heart failure."),
            "Number of other diseases": (lambda: df["comorbidity_count"].clip(upper=3).map(
                                             {0: "0", 1: "1", 2: "2", 3: "3+"}),
                                         "More other diseases means more burden and more returns to hospital."),
        }),
        "💊 Medicines": ("Descriptive", {
            "ACE inhibitor / ARB": (lambda: yes_no(drug_any(["Benazepril hydrochloride tablet", "Valsartan Dispersible tablet"]), "Given", "Not given"),
                                    "A key long-term heart medicine, given to only about 4 in 10 patients. Differences reflect "
                                    "who was well enough to receive it, not proof the drug caused them."),
            "Beta-blocker": (lambda: yes_no(drug_any(["Metoprolol Succinate Sustained-release tablet", "metoprolol tartrate injection"]), "Given", "Not given"),
                             "Another key long-term medicine given to only about 4 in 10 patients."),
            "Spironolactone": (lambda: yes_no(df["Spironolactone tablet"] == 1, "Given", "Not given"),
                               "Given to most patients. Those not given it were often too sick or had kidney problems, so this "
                               "shows a link, not proof of cause. It needs potassium checks."),
            "Water tablet by drip (IV furosemide)": (lambda: yes_no(df["Furosemide injection"] == 1, "Given", "Not given"),
                                                     "IV water tablets are used for more congested, sicker patients."),
            "Number of medicines": (lambda: cut("total_drugs", [0, 5, 9, 13, 100], ["0–4", "5–8", "9–12", "13+"]),
                                    "Patients on very few medicines had more deaths, likely because the sickest patients died or left "
                                    "before full treatment. This shows a link, not that medicines alone made the difference."),
        }),
        "🤖 Predicted Risk (model)": ("Predictive", {
            "Predicted risk group": (None,
                                     "Our Logistic Regression model scores every patient using admission data only, tested on "
                                     "patients it never saw. The highest-risk group should get closer monitoring and early follow-up."),
        }),
    }

    OUTCOMES = {"Readmission within 28 days": "re_admission_within_28_days",
                "Readmission within 3 months": "re_admission_within_3_months",
                "Readmission within 6 months": "re_admission_within_6_months",
                "Death within 28 days": "death_within_28_days",
                "Death within 3 months": "death_within_3_months",
                "Death within 6 months": "death_within_6_months"}

    # Keep the three controls on one horizontal row for presentation use.
    sel1, sel2, sel3 = st.columns(3)
    with sel1:
        area = st.selectbox("1. Select Insight Area", list(AREAS.keys()))
    kind, markers = AREAS[area]
    with sel2:
        marker = st.selectbox("2. Select Marker", list(markers.keys()))
    with sel3:
        out_label = st.selectbox("3. Select Outcome", list(OUTCOMES.keys()), index=5)
    target = OUTCOMES[out_label]
    is_death = target.startswith("death")
    make_groups, meaning = markers[marker]

    # ---------------- build the groups ----------------
    if make_groups is None:   # predicted risk groups from the model
        feats = DEATH_FEATURES if is_death else READMIT_FEATURES
        with st.spinner("Scoring patients with the model..."):
            prob = cv_probs(df, feats, target, "Logistic Regression", repeats=3)
        groups = pd.Series(pd.qcut(prob, 5, labels=["Lowest", "Low", "Middle", "High", "Highest"]), index=df.index)
        model_auc = roc_auc_score(df[target], prob)
    else:
        groups = make_groups()
        model_auc = None

    data = pd.DataFrame({"group": groups, "y": df[target]}).dropna()
    if isinstance(groups.dtype, pd.CategoricalDtype):
        order = [c for c in groups.cat.categories if c in set(data["group"])]
    else:
        order = sorted(data["group"].unique(), key=str)
    summary = data.groupby("group", observed=True)["y"].agg(["mean", "size"]).reindex(order)
    summary["rate"] = summary["mean"] * 100

    # ---------------- layout: Donut | Chart | Finding ----------------
    left, mid, right = st.columns([0.9, 1.7, 1.3])

    with left:
        yes = df[target].mean() * 100
        word = "Died" if is_death else "Came back"
        fig = go.Figure(go.Pie(values=[yes, 100 - yes], labels=[word, "Did not"], hole=0.68, sort=False,
                               marker=dict(colors=[DEATH if is_death else READMIT, "#E3ECEF"]), textinfo="none"))
        fig.update_layout(title=dict(text="All 2,008 patients", font=dict(size=14, color=NAVY)), height=300,
                          margin=dict(t=40, l=5, r=5, b=5), showlegend=True,
                          legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                          annotations=[dict(text=f"<b>{yes:.1f}%</b><br>{word.lower()}", x=0.5, y=0.5,
                                            showarrow=False, font=dict(size=18, color=NAVY))])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(out_label)

    with mid:
        accent = DEATH if is_death else READMIT
        if len(summary) > 2:
            cols_used = (RAMP * 3)[:len(summary)]
        else:   # two groups: highlight the higher-risk one
            cols_used = [accent if r == summary["rate"].max() else "#9FB7BE" for r in summary["rate"]]
        fig = px.bar(x=[str(i) for i in summary.index], y=summary["rate"], text_auto=".1f",
                     color=[str(i) for i in summary.index], color_discrete_sequence=cols_used,
                     title=f"{marker}: {out_label}")
        fig.update_traces(customdata=summary["size"], hovertemplate="%{x}<br>%{y:.1f}%<br>%{customdata} patients<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="% of patients")
        st.plotly_chart(style(fig, 360), width="stretch")

    with right:
        badge(kind)
        top, low = summary["rate"].idxmax(), summary["rate"].idxmin()
        tested = len(data)
        # Evidence: chi-square test across the groups (predicted risk uses ROC-AUC)
        if model_auc is not None:
            evidence = f"Model ROC-AUC {model_auc:.2f} on unseen patients (0.5 = coin toss)"
        elif summary["size"].min() > 0 and data["y"].nunique() == 2 and len(summary) > 1:
            _, p, _, _ = stats.chi2_contingency(pd.crosstab(data["group"], data["y"]))
            evidence = f"Chi-square test, p {'< 0.001' if p < 0.001 else '= ' + format(p, '.3f')} " \
                       f"({'significant' if p < 0.05 else 'not significant'})"
        else:
            evidence = "Descriptive comparison"
        small = summary[summary["size"] < 30]
        st.markdown(
            f"<div class='sec' style='font-size:26px'>🔎 Finding</div>"
            f"<p style='font-size:16px;color:#073B4C'>Patients in <b>{html.escape(str(top))}</b> had the highest rate: "
            f"<b>{summary.loc[top, 'rate']:.1f}%</b> ({int(summary.loc[top, 'size'])} patients), "
            f"vs <b>{summary.loc[low, 'rate']:.1f}%</b> in <b>{html.escape(str(low))}</b>. "
            f"Average for all patients: {df[target].mean()*100:.1f}%.</p>"
            f"<p style='font-size:14px;color:#637B83'><b>Evidence:</b> {evidence}. "
            f"Patients with this marker: {tested:,} of {len(df):,}."
            + (f" Small groups (under 30 patients): {html.escape(', '.join(map(str, small.index)))}." if len(small) else "")
            + "</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='todo'><b>What this means:</b> {meaning}</div>", unsafe_allow_html=True)


# =====================================================================
# 5. MODEL PERFORMANCE
# =====================================================================
elif page == "🤖 Model Performance":
    st.markdown("<div class='pagetitle'>🤖 Model Performance</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='section'>
<b>How we tested:</b> only information available <b>at admission</b> was used. Each model was trained on 4/5 of the
patients and tested on the other 1/5, five times over (5-fold cross-validation), so every score comes from unseen patients.
<br><br>
<b>How to read the scores:</b>
<ul style='margin-bottom:0'>
<li><b>ROC-AUC</b>: how often the model ranks a patient who had the outcome above one who did not. 0.5 = coin toss, 1.0 = perfect.</li>
<li><b>Recall</b>: of the patients who had the outcome, how many the model flagged.</li>
<li><b>Precision</b>: of the patients the model flagged, how many really had the outcome.</li>
<li><b>Accuracy is misleading here:</b> only 3% die, so a model that says "nobody dies" is 97% accurate and useless.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    targets = {"6-month death": ("death_within_6_months", DEATH_FEATURES, df),
               "28-day death": ("death_within_28_days", DEATH_FEATURES, df),
               "6-month readmission": ("re_admission_within_6_months", READMIT_FEATURES,
                                       df[(df["outcome_during_hospitalization"] != "Dead") &
                                          (df["death_within_6_months"] == 0)].reset_index(drop=True))}
    choice = st.selectbox("Outcome to predict", list(targets.keys()))
    target, feats, data = targets[choice]
    y = data[target].values

    with st.spinner("Training and testing 3 models..."):
        rows, curves, preds = [], {}, {}
        for name in model_set():
            p = cv_probs(data, feats, target, name)
            pred = (p >= 0.5).astype(int)
            preds[name] = pred
            curves[name] = roc_curve(y, p)
            rows.append([name, roc_auc_score(y, p), recall_score(y, pred, zero_division=0),
                         precision_score(y, pred, zero_division=0), accuracy_score(y, pred)])
        rows.append(["Baseline: predict 'no' for everyone", 0.5, 0.0, 0.0, 1 - y.mean()])
    res = pd.DataFrame(rows, columns=["Model", "ROC-AUC", "Recall", "Precision", "Accuracy"])

    lr = res.set_index("Model").loc["Logistic Regression"]
    ann = res.set_index("Model").loc["ANN (neural network)"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🏆", "Chosen model", "Logistic Regression", size=20)
    with c2: kpi("📈", "ROC-AUC (chosen model)", f"{lr['ROC-AUC']:.2f}")
    with c3: kpi("🎯", "Patients caught (recall)", pct(lr["Recall"]))
    with c4: kpi("👥", "Patients with outcome", f"{int(y.sum())} of {len(y):,}")
    st.write("")

    st.dataframe(res.style.format({c: "{:.2f}" for c in ["ROC-AUC", "Recall", "Precision", "Accuracy"]}),
                 hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, colr in zip(curves, [GREEN, BLUE, NAVY]):
            fpr, tpr, _ = curves[name]
            auc = res.set_index("Model").loc[name, "ROC-AUC"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.2f})", line=dict(color=colr, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin toss", line=dict(color=ALERT, dash="dash")))
        fig.update_layout(title="ROC curve (higher and more to the left = better)",
                          xaxis_title="False alarms (rate)", yaxis_title="Patients caught (rate)")
        st.plotly_chart(style(fig, 400), width="stretch")
    with right:
        cm = confusion_matrix(y, preds["Logistic Regression"])
        cm_df = pd.DataFrame(cm, index=["Actual: no", "Actual: yes"], columns=["Flagged: no", "Flagged: yes"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=["#EAF5F8", TEAL2, NAVY],
                        title="Logistic Regression: who it flagged")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 400), width="stretch")

    found(f"All three models <b>rank</b> patients about equally well (ROC-AUC {res['ROC-AUC'][:3].min():.2f} to "
          f"{res['ROC-AUC'][:3].max():.2f}). The difference is who they actually <b>flag</b>: Logistic Regression catches "
          f"<b>{lr['Recall']*100:.0f}%</b> of these patients, while the neural network catches {ann['Recall']*100:.0f}% "
          f"and can look 'accurate' only because it says 'no' to almost everyone, like the baseline row. "
          f"So we chose <b>Logistic Regression</b>: it catches the most high-risk patients and is easy to explain to doctors.")
    todo("A flagged patient is not a diagnosis. Flags point the team to who needs a closer look first.")

    # ---------------- Key models at a glance (scores calculated live, on unseen patients) ----------------
    st.subheader("Our key models at a glance")
    alive6 = targets["6-month readmission"][2]
    glance = [
        ("Death within 28 days", "Bedside check only (Killip + NYHA)",
         df, ["killip_grade", "nyha_cardiac_function_classification"], "death_within_28_days", "Very good with just a 30-second exam"),
        ("Death within 6 months", "Bedside check + 6 routine blood tests",
         df, DEATH_FEATURES, "death_within_6_months", "Best overall death model"),
        ("Death within 28 days", "NLR from the routine blood count",
         df, ["nlr_log"], "death_within_28_days", "A free test with useful signal"),
        ("Came back within 6 months", "21 admission measures",
         alive6, READMIT_FEATURES, "re_admission_within_6_months", "Weak, but top-risk group returns 2x as often"),
    ]
    summary = pd.DataFrame({
        "What we predicted": [g[0] for g in glance],
        "Using": [g[1] for g in glance],
        "ROC-AUC": [f"{roc_auc_score(g[2][g[4]], cv_probs(g[2], g[3], g[4], 'Logistic Regression', repeats=3)):.2f}"
                    for g in glance],
        "In simple words": [g[5] for g in glance]})
    st.dataframe(summary, hide_index=True, width="stretch")

    # ---------------- Patient risk check ----------------
    st.subheader("🩺 Try it: Patient Risk Check")
    st.caption("Uses the 6-month death model. Pick a real patient or enter a new one. "
               "It supports the doctor's judgement; it does not replace it.")

    @st.cache_resource
    def final_model():
        model = logistic().fit(df[DEATH_FEATURES].astype(float), df["death_within_6_months"])
        prob = cv_probs(df, DEATH_FEATURES, "death_within_6_months", "Logistic Regression", repeats=3)
        edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
        rate = pd.Series(df["death_within_6_months"].values).groupby(np.digitize(prob, edges)).mean() * 100
        return model, edges, rate, prob

    model, edges, death_rate, cv_prob = final_model()

    ids = ["New patient (enter values)"] + sorted(df["inpatient_number"].astype(int).tolist())
    pid = st.selectbox("Patient ID", ids, help="Pick a patient from our data to fill in their admission values, "
                                                "or choose 'New patient' and type the values.")
    med = df[["brain_natriuretic_peptide", "high_sensitivity_troponin", "neutrophil_count", "lymphocyte_count",
              "albumin", "hemoglobin", "sodium", "glomerular_filtration_rate", "systolic_blood_pressure"]].median()
    if pid == ids[0]:
        d = {"nyha": 3, "killip": 2, "bnp": 750.0, "trop": 55.0, "neut": 5.0, "lymph": 1.0,
             "alb": 37.0, "hb": 115.0, "na": 139.0, "egfr": 60.0, "sbp": 130.0}
        row_i = None
    else:
        row_i = df.index[df["inpatient_number"].astype(int) == pid][0]
        r = df.loc[row_i]

        def val(col, lo, hi):
            v = r[col] if pd.notna(r[col]) else med[col]
            return float(min(max(v, lo), hi))

        d = {"nyha": int(r["nyha_cardiac_function_classification"]), "killip": int(r["killip_grade"]),
             "bnp": val("brain_natriuretic_peptide", 10, 5000), "trop": val("high_sensitivity_troponin", 0, 50000),
             "neut": val("neutrophil_count", 0.1, 50), "lymph": val("lymphocyte_count", 0.05, 20),
             "alb": val("albumin", 10, 60), "hb": val("hemoglobin", 30, 200), "na": val("sodium", 110, 160),
             "egfr": val("glomerular_filtration_rate", 1, 200), "sbp": val("systolic_blood_pressure", 50, 250)}

    with st.form(f"patient_{pid}"):
        c1, c2, c3, c4 = st.columns(4)
        nyha = c1.selectbox("NYHA class (symptoms)", [1, 2, 3, 4], index=[1, 2, 3, 4].index(d["nyha"]))
        killip = c2.selectbox("Killip grade (fluid / shock)", [1, 2, 3, 4], index=d["killip"] - 1)
        bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, d["bnp"])
        trop = c4.number_input("Troponin (pg/mL)", 0.0, 50000.0, d["trop"])
        c5, c6, c7, c8 = st.columns(4)
        neut = c5.number_input("Neutrophils (x10^9/L)", 0.1, 50.0, d["neut"])
        lymph = c6.number_input("Lymphocytes (x10^9/L)", 0.05, 20.0, d["lymph"])
        alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, d["alb"])
        hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, d["hb"])
        c9, c10, c11, _ = st.columns(4)
        na = c9.number_input("Sodium (mmol/L)", 110.0, 160.0, d["na"])
        egfr = c10.number_input("eGFR (kidney)", 1.0, 200.0, d["egfr"])
        sbp_in = c11.number_input("Systolic BP (mmHg)", 50.0, 250.0, d["sbp"])
        submitted = st.form_submit_button("Check risk", type="primary")

    if submitted:
        nlr_val = neut / lymph
        model_inputs = {"nyha": nyha, "killip": killip, "bnp": bnp, "trop": trop, "neut": neut, "lymph": lymph,
                        "alb": alb, "hb": hbv, "na": na}
        if row_i is not None and all(model_inputs[k] == d[k] for k in model_inputs):
            # Existing patient, model values unchanged: use the risk from a model that never saw this patient
            score = cv_prob[row_i]
        else:
            x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(nlr_val), alb, hbv, na]],
                             columns=DEATH_FEATURES)
            score = model.predict_proba(x)[0, 1]
        grp = int(np.digitize(score, edges))
        names = ["Lowest", "Low", "Middle", "High", "Highest"]
        colours = [RAMP[0], RAMP[1], "#F2C14E", "#E07A5F", ALERT]
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Risk group: <span style='color:{colours[grp]}'>{names[grp]}</span>", unsafe_allow_html=True)
            kpi("⚠️", "Similar patients who died within 6 months", f"{death_rate.iloc[grp]:.1f}%")
            if row_i is not None:
                died = df.loc[row_i, "death_within_6_months"] == 1
                back = df.loc[row_i, "re_admission_within_6_months"] == 1
                st.markdown(f"<div class='found'><b>What really happened to patient {pid}:</b><br>"
                            f"Died within 6 months: <b>{'Yes' if died else 'No'}</b><br>"
                            f"Came back within 6 months: <b>{'Yes' if back else 'No'}</b></div>",
                            unsafe_allow_html=True)
            flags = [("Fluid in lungs or shock (Killip 3-4)", killip >= 3), ("Symptoms at rest (NYHA IV)", nyha == 4),
                     ("Low blood pressure (below 90)", sbp_in < 90), ("Weak kidneys (eGFR below 45)", egfr < 45),
                     ("Severe anemia (hemoglobin below 80)", hbv < 80), (f"High NLR ({nlr_val:.1f})", nlr_val >= 8.7)]
            shown = [n for n, on in flags if on]
            st.write("")
            st.markdown("**Warning signs:**")
            for n in shown:
                st.error(n)
            if not shown:
                st.success("No warning signs.")
        with right:
            st.plotly_chart(bar(names, death_rate.values, "Deaths within 6 months by risk group (%)", colours, height=320),
                            width="stretch")


# =====================================================================
# 6. KEY TAKEAWAYS & CONCLUSION
# =====================================================================
elif page == "📌 Key Takeaways & Conclusion":
    st.markdown("<div class='pagetitle'>📌 Key Takeaways</div>", unsafe_allow_html=True)
    take = ["Coming back to hospital (38.5% in 6 months) is a much bigger problem than death (2.8%)",
            "How sick the patient is today matters most: 27% of Killip 4 patients died within 6 months vs 0.8% of Killip 1",
            "Heart and organ warning signs: very high troponin, weak kidneys, high potassium, severe anemia and low sodium",
            "Simple routine tests work best: NLR is free and available for 99% of patients, while hs-CRP and blood gas were missing for half",
            "Only about 4 in 10 patients get the key long-term heart medicines (ACE inhibitor/ARB, beta-blocker)",
            "Our simple, explainable model (Logistic Regression) catches 70% of 6-month deaths; the neural network caught none"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in take)
    st.markdown(f"<div class='checkbox'><b class='h'>Key Clinical Findings:</b>{items}</div>", unsafe_allow_html=True)

    st.markdown("<div class='pagetitle' style='font-size:36px'>🏁 Conclusion</div>", unsafe_allow_html=True)
    concl = ["With tests the hospital already does on day 1 (bedside check + routine blood tests), it can spot high-risk patients early",
             "Acting on these warning signs can save lives, free ICU beds and reduce readmissions through early follow-up",
             "Limits: one hospital's data and few deaths; results show links, not proof of cause"]
    items = "".join(f"<div class='it'>✅ {x}</div>" for x in concl)
    st.markdown(f"<div class='checkbox'>{items}</div>", unsafe_allow_html=True)
