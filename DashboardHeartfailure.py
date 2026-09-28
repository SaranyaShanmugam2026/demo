# =====================================================================
#  Cardiac Failure Analytics Dashboard
#  Team Python Pioneers | NumpyNinja Python Hackathon
#
#  Run:  streamlit run DashboardHeartfailure.py
#  Data: Cardiac_Cleaned_Data.xlsb (or Cardiac_Cleaned_Data.csv) in the same folder
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

st.set_page_config(page_title="Cardiac Failure Analytics", page_icon="❤️", layout="wide")

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

/* Sidebar navigation as buttons (like the diabetes dashboard) */
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
</style>
""", unsafe_allow_html=True)


# ----------------------------- SMALL HELPERS -----------------------------
def kpi(icon, title, value, size=26):
    st.markdown(f"<div class='kpi'><div class='i'>{icon}</div><div class='t'>{title}</div>"
                f"<div class='v' style='font-size:{size}px'>{value}</div></div>", unsafe_allow_html=True)


def found(text):
    st.markdown(f"<div class='found'><b>What we found:</b> {text}</div>", unsafe_allow_html=True)


def todo(text):
    st.markdown(f"<div class='todo'><b>What it means for the hospital:</b> {text}</div>", unsafe_allow_html=True)


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
                 color_discrete_map={"Readmitted in 6 months": READMIT, "Died in 6 months": DEATH}, title=title)
    fig.update_layout(yaxis_title="% of patients", xaxis_title="")
    return style(fig, 400)


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------- DATA -----------------------------
HERE = Path(__file__).parent


@st.cache_data
def load_data():
    xlsb, csv = HERE / "Cardiac_Cleaned_Data.xlsb", HERE / "Cardiac_Cleaned_Data.csv"
    if xlsb.exists():
        df = pd.read_excel(xlsb, engine="pyxlsb")
    else:
        df = pd.read_csv(csv)
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

# Descriptive demographic columns used in the Data Overview section.
def find_col(names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

gender_col = find_col(["gender", "sex"])
agecat_col = find_col(["agecat", "age_category"])
bmi_col = find_col(["bmi"])

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
    st.markdown("<div style='text-align:center;font-size:48px'>❤️</div>"
                "<h2 style='text-align:center;margin:0'>HeartFailure</h2>"
                "<p style='text-align:center'>Team Python Pioneers</p>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["🏠 Introduction", "📘 Data Overview", "🧹 Data Cleaning & Features",
                                   "🩺 Interactive Clinical Insights", "🤖 Model Performance", "📌 Key Takeaways & Conclusion"],
                    label_visibility="collapsed")


# =====================================================================
# 1. INTRODUCTION
# =====================================================================
if page == "🏠 Introduction":
    st.markdown("<div class='hdr'><h1>❤️ HeartFailure Clinical Explorer</h1><p>Heart-failure data analytics, outcomes and risk exploration</p></div>", unsafe_allow_html=True)

    intro_img = Path(__file__).parent / "heartfailure_intro.png"
    if intro_img.exists():
        st.image(str(intro_img), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='member'><div style='font-size:32px'>📌</div><div class='n'>Project</div><div class='r'>HeartFailure Clinical Explorer</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='member'><div style='font-size:32px'>👥</div><div class='n'>Team</div><div class='r'>Python Pioneers</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='member'><div style='font-size:32px'>🧑‍💻</div><div class='n'>Team Members</div><div class='r'>Aditi Mishra • Saranya Shanmugam • Sahi Laguduva • Sudha madhuri Basa</div></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section' style='margin-top:18px'>
    <h3 style='color:#073B4C;margin-top:0'>Project Focus</h3>
    <p style='font-size:17px;line-height:1.7;margin-bottom:0'>
    The <b>HeartFailure Clinical Explorer</b> organizes patient, cardiac, laboratory, history, hospitalization and outcome information into an interactive analytics dashboard for understanding clinical patterns and mortality/readmission outcomes.
    </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# 2. DATA OVERVIEW
# =====================================================================
elif page == "📘 Data Overview":
    st.markdown("<div class='hdr'><h1>📘 HeartFailure Data Overview</h1><p>Understanding the source dataset, the patient information it contains, and how our project uses it</p></div>", unsafe_allow_html=True)

    overview_img = Path(__file__).parent / "heartfailure_data_overview.png"
    if overview_img.exists():
        st.image(str(overview_img), use_container_width=True)

    project_patients = int(df["inpatient_number"].nunique()) if "inpatient_number" in df.columns else len(df)
    st.markdown("""
    <div class='section'>
    <h3 style='color:#073B4C;margin-top:0'>What is the HeartFailure dataset?</h3>
    <p style='font-size:16px;line-height:1.7'>
    The <b>HeartFailure dataset</b> used in this project comes from the PhysioNet resource
    <b>“Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data.”</b>
    It is a retrospective hospital dataset containing <b>2,008 patients and 168 variables</b>. The patients were admitted with heart failure at
    <b>Zigong Fourth People's Hospital, Sichuan, China</b>, and the source study covers <b>December 2016 through June 2019</b>.
    </p>
    <p style='font-size:16px;line-height:1.7;margin-bottom:0'>
    The source data combines information recorded around hospitalization with follow-up information collected at
    <b>28 days, 3 months and 6 months</b>. This allows the project to examine both the patient's clinical profile at admission and later outcomes such as mortality and readmission.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("👥", "Original patients", "2,008")
    with c2: kpi("📋", "Original variables", "168")
    with c3: kpi("🗂️", "Source tables integrated", "7")
    with c4: kpi("👤", "Patients in project", f"{project_patients:,}")
    with c5: kpi("📅", "Follow-up", "28d • 3m • 6m", size=20)

    st.markdown("""
    <div class='section'>
    <h3 style='color:#073B4C;margin-top:0'>What does the dataset contain?</h3>
    <p style='line-height:1.6'>The HeartFailure data covers several complementary clinical domains. Together, these domains provide a patient-level view rather than relying on a single biomarker or diagnosis field.</p>
    <table style='width:100%;border-collapse:collapse;font-size:15px'>
    <tr><th style='text-align:left;padding:11px;border-bottom:2px solid #D9E7EB'>Clinical domain</th><th style='text-align:left;padding:11px;border-bottom:2px solid #D9E7EB'>Examples in this project</th><th style='text-align:left;padding:11px;border-bottom:2px solid #D9E7EB'>Role in the analysis</th></tr>
    <tr><td style='padding:11px'><b>Patient profile</b></td><td style='padding:11px'>Gender, age category, height, weight, BMI, occupation</td><td style='padding:11px'>Describes who is represented in the hospitalized cohort</td></tr>
    <tr><td style='padding:11px'><b>Cardiac severity</b></td><td style='padding:11px'>NYHA, Killip, LVEF, LVEDD, E/A, valve measures</td><td style='padding:11px'>Characterizes heart-failure severity and cardiac function</td></tr>
    <tr><td style='padding:11px'><b>Medical history</b></td><td style='padding:11px'>Prior myocardial infarction, vascular disease, diabetes, COPD and comorbidity measures</td><td style='padding:11px'>Adds prior disease burden and clinical context</td></tr>
    <tr><td style='padding:11px'><b>Laboratory & biomarkers</b></td><td style='padding:11px'>BNP, troponin, creatinine, eGFR, urea, albumin, hemoglobin, electrolytes, inflammation markers</td><td style='padding:11px'>Represents cardiac injury, kidney function, nutrition, blood status and systemic stress</td></tr>
    <tr><td style='padding:11px'><b>Hospital episode</b></td><td style='padding:11px'>Admission type, length of stay, respiratory support, oxygen use, discharge information</td><td style='padding:11px'>Describes the hospitalization and disposition</td></tr>
    <tr><td style='padding:11px'><b>Medications</b></td><td style='padding:11px'>Medication indicators created from prescription records</td><td style='padding:11px'>Represents medication exposure during the hospital episode</td></tr>
    <tr><td style='padding:11px'><b>Outcomes</b></td><td style='padding:11px'>In-hospital status, 28-day/3-month/6-month mortality and readmission, emergency return</td><td style='padding:11px'>Provides the endpoints examined by the descriptive and predictive analyses</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------- DEMOGRAPHIC PROFILE -----------------------------
    st.markdown("""
    <div class='section'>
    <h3 style='color:#073B4C;margin-top:0'>👤 Patient Demographic Profile</h3>
    <p style='line-height:1.6;margin-bottom:10px'>This descriptive profile shows who is represented in the HeartFailure cohort before clinical severity and outcome analysis. It summarizes gender, source age category, BMI and occupation without treating these distributions as clinical recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        if gender_col and gender_col in df.columns:
            g = df[gender_col].dropna().astype(str).str.strip().value_counts().reset_index()
            g.columns = ["Gender", "Patients"]
            fig = px.pie(g, names="Gender", values="Patients", hole=0.52, title="Gender distribution", color_discrete_sequence=[TEAL2, BLUE, GREEN])
            fig.update_layout(height=320, legend_title="", margin=dict(t=55,l=10,r=10,b=10))
            st.plotly_chart(style(fig, 320), width="stretch")
        else:
            st.info("Gender data is not available in the current dataset.")
    with d2:
        if agecat_col and agecat_col in df.columns:
            a = df[agecat_col].dropna().astype(str).str.strip().value_counts().reset_index()
            a.columns = ["Age category", "Patients"]
            fig = px.bar(a, x="Age category", y="Patients", text_auto=True, title="Age-category distribution", color_discrete_sequence=[TEAL2])
            fig.update_layout(height=320, xaxis_title="Age category", yaxis_title="Patients")
            st.plotly_chart(style(fig, 320), width="stretch")
        else:
            st.info("Age-category data is not available in the current dataset.")

    d3, d4 = st.columns(2)
    with d3:
        if bmi_col and bmi_col in df.columns:
            bmi = pd.to_numeric(df[bmi_col], errors="coerce").dropna()
            if len(bmi):
                fig = px.histogram(x=bmi, nbins=25, title="BMI distribution", labels={"x":"BMI", "y":"Patients"}, color_discrete_sequence=[GREEN])
                fig.update_layout(height=300, bargap=0.05)
                st.plotly_chart(style(fig, 300), width="stretch")
        else:
            st.info("BMI data is not available in the current dataset.")
    with d4:
        if "occupation" in df.columns:
            occ = df["occupation"].dropna().astype(str).str.strip().value_counts().head(8).reset_index()
            occ.columns = ["Occupation", "Patients"]
            fig = px.bar(occ, x="Patients", y="Occupation", orientation="h", text_auto=True, title="Most common occupation groups", color_discrete_sequence=[BLUE])
            fig.update_layout(height=300, yaxis_title="", xaxis_title="Patients")
            st.plotly_chart(style(fig, 300), width="stretch")
        else:
            st.info("Occupation data is not available in the current dataset.")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("""
        <div class='section'>
        <h3 style='color:#073B4C;margin-top:0'>From the source dataset to our project dataset</h3>
        <div class='found'><b>1. Integrate</b><br>Seven source tables are linked through <b>inpatient_number</b> to create a patient-level analytical dataset.</div>
        <div class='found'><b>2. Clean</b><br>Invalid or inconsistent values are reviewed, missingness is assessed, and variables are standardized for analysis.</div>
        <div class='found'><b>3. Transform</b><br>Prescription records are converted into patient-level medication indicators and clinically interpretable features are engineered.</div>
        <div class='found'><b>4. Analyze</b><br>Descriptive and prescriptive analyses examine severity, biomarkers, comorbidities, mortality, readmission and emergency-return patterns.</div>
        <div class='todo'><b>5. Model</b><br>Logistic Regression and Random Forest are used as comparison models alongside the project's primary Artificial Neural Network analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        out_cols = ["re_admission_within_28_days","re_admission_within_3_months","re_admission_within_6_months","death_within_28_days","death_within_3_months","death_within_6_months"]
        available = [c for c in out_cols if c in df.columns]
        if available:
            rows=[]
            labels={"re_admission_within_28_days":"Readmission — 28 days","re_admission_within_3_months":"Readmission — 3 months","re_admission_within_6_months":"Readmission — 6 months","death_within_28_days":"Mortality — 28 days","death_within_3_months":"Mortality — 3 months","death_within_6_months":"Mortality — 6 months"}
            for c in available: rows.append((labels[c], float(pd.to_numeric(df[c], errors="coerce").mean()*100)))
            odf=pd.DataFrame(rows,columns=["Outcome","Percent"])
            fig=px.bar(odf,x="Percent",y="Outcome",orientation="h",text_auto=".1f",color_discrete_sequence=[TEAL2],title="Observed outcome rates in our project data")
            fig.update_layout(xaxis_title="Patients (%)",yaxis_title="",height=360)
            st.plotly_chart(style(fig),width="stretch")
        st.markdown("""
        <div class='section'>
        <h4 style='color:#073B4C;margin-top:0'>Why this dataset fits the project</h4>
        <p style='line-height:1.6;margin-bottom:8px'>The dataset links <b>admission severity</b>, <b>cardiac function</b>, <b>laboratory markers</b>, <b>comorbidity</b> and <b>follow-up outcomes</b>. That combination supports the project's central question: whether multiple patient characteristics can be combined to identify patterns associated with mortality and other adverse outcomes.</p>
        <p style='line-height:1.6;margin-bottom:0'><b>Important:</b> this is a retrospective, single-center dataset. The source documentation notes that the data are aggregated at the hospitalization level and do not provide time-series measurements throughout the stay, so model results should be treated as analytical findings rather than clinically validated decision rules.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Source: PhysioNet, HeartFailure dataset version 1.3. The source describes 2,008 patients, 168 variables, the December 2016–June 2019 study period, and follow-up at 28 days, 3 months and 6 months.")
    st.link_button("Open the official PhysioNet HeartFailure dataset description", "https://www.physionet.org/content/heart-failure-zigong/1.3/")

# =====================================================================
# 3. DATA CLEANING & FEATURE ENGINEERING
# =====================================================================
elif page == "🧹 Data Cleaning & Features":
    st.markdown("<div class='hdr'><h1>🧹 Data Cleaning & Feature Engineering</h1>"
                "<p>From 7 messy tables to 1 trusted table (one row per patient)</p></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🗂️", "Tables joined", "7 → 1")
    with c2: kpi("🧽", "Cleaning steps", "27")
    with c3: kpi("✨", "New features", "16")
    with c4: kpi("🆔", "Duplicate patients", "0")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("""
<div class='section'>
<h4 style='color:#073B4C;margin-top:0'>🧹 What we cleaned</h4>
<ul>
<li><b>Removed impossible values:</b> a fake patient, 0 kg weight, 0 pulse, BMI of 404.</li>
<li><b>Fixed wrong units:</b> troponin, hematocrit and heart-scan values.</li>
<li><b>Filled blanks only when the meaning was clear:</b> blank breathing support = no ventilation.</li>
<li><b>Kept real gaps empty:</b> we did not invent missing lab results.</li>
<li><b>Medicines:</b> changed from many rows per patient to one row per patient.</li>
<li><b>Joined all 7 tables</b> into one table: 2,008 patients.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section'><h4 style='color:#073B4C;margin-top:0'>✅ Why it mattered</h4>", unsafe_allow_html=True)
        ba = pd.DataFrame({
            "Example": ["Patients showing heart damage", "Highest BMI", "Rows per patient (medicines)", "Patients with E/A ratio"],
            "Before": ["0.3% (wrong unit)", "404 (impossible)", "about 8", "393"],
            "After": ["84% (as expected)", "39", "1", "536 (calculated)"]})
        st.dataframe(ba, hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
<div class='section'>
<h4 style='color:#073B4C;margin-top:0'>✨ Feature engineering: new columns we added, and why</h4>

| New feature | Why we added it |
|---|---|
| **Groups** (BMI group, blood pressure group, kidney stage, anemia level) | Easier to compare and explain than raw numbers |
| **Yes/No warning flags** (high BNP, high troponin, enlarged heart, many medicines) | Each one answers a simple clinical question |
| **Number of other diseases** | One number for how much extra illness a patient carries |
| **NLR** (neutrophil ÷ lymphocyte) | A free inflammation marker from the routine blood count |
| **Log of skewed values** (BNP, troponin) | Stops a few extreme patients from controlling the model |
</div>
""", unsafe_allow_html=True)

    col = st.selectbox("See how one new feature splits the patients",
                       ["bmi_category", "bp_category", "gcs_category", "medication_burden"],
                       format_func=lambda c: {"bmi_category": "BMI group", "bp_category": "Blood pressure group",
                                              "gcs_category": "Alertness group (GCS)", "medication_burden": "Medicine load"}[c])
    counts = df[col].value_counts().rename_axis("Group").reset_index(name="Patients")
    st.plotly_chart(bar(counts["Group"].astype(str), counts["Patients"], "", [TEAL2, BLUE, GREEN, NAVY, TEAL],
                        ytitle="Patients", fmt=",", height=320), width="stretch")


# =====================================================================
# 4. INSIGHTS (descriptive + prescriptive + predictive in tabs)
# =====================================================================
elif page == "🩺 Interactive Clinical Insights":
    st.markdown("<div class='hdr'><h1>🩺 Interactive Clinical Insights</h1><p>Select an analysis area, outcome and clinical marker to see the finding, evidence and interpretation.</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section'>
    <h3 style='color:#073B4C;margin-top:0'>🔍 How to use this page</h3>
    <p>Choose <b>what clinical area you want to investigate</b>, then select the <b>marker</b> and the <b>outcome</b>. The dashboard updates the chart and the written finding automatically.</p>
    <p style='margin-bottom:0;color:#637B83'>This makes each insight traceable: <b>Question → Evidence → Finding → Interpretation</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    areas = [
        "🫘 Kidney Function",
        "🩸 Inflammation & Nutrition",
        "❤️ Cardiac Biomarkers",
        "🩺 Current Clinical Severity",
        "🧭 Current Severity vs Prior History",
        "🫁 Blood Gas",
        "🩸 Anemia",
        "🔁 Readmission Patterns",
    ]
    area = st.selectbox("1. Select Insight Area", areas)

    def outcome_series(label):
        mapping = {
            "28-Day Mortality": "death_within_28_days",
            "6-Month Mortality": "death_within_6_months",
            "6-Month Readmission": "re_admission_within_6_months",
            "In-Hospital Mortality": "in_hospital_death",
        }
        return mapping[label]

    # Build a safe in-hospital mortality proxy from the hospitalization outcome.
    insight_df = df.copy()
    if "in_hospital_death" not in insight_df.columns:
        insight_df["in_hospital_death"] = (insight_df["outcome_during_hospitalization"].astype(str).str.strip().str.lower() == "dead").astype(int)

    marker = None
    outcome_label = None
    target = None
    table = None
    chart_title = ""
    finding = ""
    interpretation = ""
    talk_track = ""
    p_text = None
    chart = None

    # ---------------- Kidney ----------------
    if area == "🫘 Kidney Function":
        marker_options = {
            "eGFR": ("glomerular_filtration_rate", "Abnormal: eGFR < 60", lambda x: x < 60,
                     {"6-Month Readmission": ("eGFR < 60", "eGFR ≥ 60", 44.8, 33.6, "p < 0.001"),
                      "6-Month Mortality": ("eGFR < 60", "eGFR ≥ 60", 4.0, 2.0, "p = 0.014")} ),
            "Creatinine": ("creatinine_enzymatic_method", "Abnormal: creatinine > 110", lambda x: x > 110,
                     {"6-Month Readmission": ("Creatinine > 110", "Creatinine ≤ 110", 46.2, 35.2, "p < 0.001"),
                      "6-Month Mortality": ("Creatinine > 110", "Creatinine ≤ 110", 4.9, 1.9, "p < 0.001")} ),
            "Urea": ("urea", "Abnormal: urea > 8.3", lambda x: x > 8.3,
                     {"6-Month Readmission": ("Urea > 8.3", "Urea ≤ 8.3", 42.6, 35.1, "p = 0.001"),
                      "6-Month Mortality": ("Urea > 8.3", "Urea ≤ 8.3", 4.0, 1.7, "p = 0.003")} ),
            "Cystatin C": ("cystatin", "Abnormal: cystatin > 0.98", lambda x: x > 0.98,
                     {"6-Month Readmission": ("Cystatin > 0.98", "Cystatin ≤ 0.98", None, None, "p = 0.226"),
                      "6-Month Mortality": ("Cystatin > 0.98", "Cystatin ≤ 0.98", None, None, "p = 0.056")} ),
        }
        marker = st.selectbox("2. Select Kidney Marker", list(marker_options.keys()))
        outcome_label = st.selectbox("3. Select Outcome", ["6-Month Readmission", "6-Month Mortality"])
        col, _, _, pmap = marker_options[marker]
        target = outcome_series(outcome_label)
        sub = insight_df.dropna(subset=[col, target]).copy()
        high = pmap[outcome_label]
        if high[2] is not None:
            labels = [high[0], high[1]]
            vals = [high[2], high[3]]
            table = pd.DataFrame({"Group": labels, "Outcome rate (%)": vals})
            chart = px.bar(table, x="Group", y="Outcome rate (%)", text_auto=".1f", color="Group",
                           color_discrete_sequence=[DEATH if "Mortality" in outcome_label else READMIT, "#9FB7BE"],
                           title=f"{marker}: {outcome_label}")
            chart.update_layout(showlegend=False, yaxis_title="% of patients", xaxis_title="")
            if vals[0] > vals[1]:
                difference = vals[0] - vals[1]
                finding = f"Patients with abnormal {marker.lower()} had a higher observed {outcome_label.lower()} rate: <b>{vals[0]:.1f}%</b> versus <b>{vals[1]:.1f}%</b>, a difference of <b>{difference:.1f} percentage points</b>."
            else:
                finding = f"The observed difference for abnormal {marker.lower()} was small in this analysis: <b>{vals[0]:.1f}%</b> versus <b>{vals[1]:.1f}%</b>."
        else:
            # Cystatin C is intentionally shown as a neutral comparison because the source analysis did not find a clear separation.
            vals = sub.groupby((sub[col] > 0.98).map({True:"Cystatin > 0.98", False:"Cystatin ≤ 0.98"}), observed=False)[target].mean().mul(100)
            table = vals.rename("Outcome rate (%)").reset_index().rename(columns={"index":"Group"})
            chart = px.bar(table, x="Group", y="Outcome rate (%)", text_auto=".1f", color="Group",
                           color_discrete_sequence=[TEAL2, "#9FB7BE"], title=f"{marker}: {outcome_label}")
            chart.update_layout(showlegend=False, yaxis_title="% of patients", xaxis_title="")
            finding = f"Cystatin C was above the normal limit in most patients, and the source analysis did not show a statistically clear separation for {outcome_label.lower()} (<b>{high[4]}</b>)."
        p_text = high[4]
        interpretation = "In this dataset, kidney-function markers provide useful context for longer-term outcomes, with creatinine showing the largest observed difference among the four markers examined. These are associations, not proof that the marker caused the outcome."
        talk_track = f"I selected Kidney Function, then {marker}, then {outcome_label}. The chart compares patients above and below the study threshold. The key point is that the observed outcome rate is higher in the abnormal group for this marker, and the p-value shown comes directly from our prescriptive analysis."

    # ---------------- Inflammation + albumin ----------------
    elif area == "🩸 Inflammation & Nutrition":
        marker = st.selectbox("2. Select Inflammation/Nutrition View", ["Inflammation + Albumin Group", "NLR", "WBC", "hs-CRP", "Albumin"])
        outcome_label = st.selectbox("3. Select Outcome", ["28-Day Mortality", "6-Month Mortality", "6-Month Readmission"])
        target = outcome_series(outcome_label)
        q = insight_df.dropna(subset=["albumin"]).copy()
        q["nlr_calc"] = q["neutrophil_count"] / q["lymphocyte_count"]
        q["inflamed"] = ((q["hs_crp"] > 5) | (q["white_blood_cell"] > 10) | (q["nlr_calc"] > 6)).astype(int)
        q["low_albumin"] = (q["albumin"] < 35).astype(int)
        q["Group"] = np.select([ (q["inflamed"]==1)&(q["low_albumin"]==1), q["inflamed"]==1, q["low_albumin"]==1], ["Both", "Inflamed only", "Low albumin only"], default="Neither")
        if marker == "Inflammation + Albumin Group":
            table = q.groupby("Group", observed=True)[target].mean().mul(100).reindex(["Neither","Inflamed only","Low albumin only","Both"]).reset_index()
            table.columns = ["Group","Outcome rate (%)"]
            chart = px.bar(table, x="Group", y="Outcome rate (%)", text_auto=".1f", color="Group", color_discrete_sequence=["#9FB7BE", TEAL2, "#6CC3B0", DEATH], title=f"Inflammation + Albumin: {outcome_label}")
            chart.update_layout(showlegend=False, yaxis_title="% of patients", xaxis_title="")
            if outcome_label == "28-Day Mortality":
                p_text = "p < 0.001"
            elif outcome_label == "6-Month Mortality":
                p_text = "p = 0.002 (Both vs Neither)"
            else:
                p_text = "Descriptive comparison"
            both_rate = float(table.loc[table["Group"]=="Both","Outcome rate (%)"].iloc[0])
            neither_rate = float(table.loc[table["Group"]=="Neither","Outcome rate (%)"].iloc[0])
            finding = f"Patients with both inflammation and low albumin had an observed {outcome_label.lower()} rate of <b>{both_rate:.1f}%</b>, compared with <b>{neither_rate:.1f}%</b> in the Neither group."
        else:
            spec = {"NLR":"nlr_calc", "WBC":"white_blood_cell", "hs-CRP":"hs_crp", "Albumin":"albumin"}[marker]
            q2=q.dropna(subset=[spec,target]).copy()
            q2["Group"] = pd.qcut(q2[spec],4,labels=["Lowest","Low","High","Highest"] if marker != "Albumin" else ["Lowest","Low","High","Highest"], duplicates="drop")
            table=q2.groupby("Group",observed=True)[target].mean().mul(100).reset_index(); table.columns=["Group","Outcome rate (%)"]
            chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".1f",color="Group",color_discrete_sequence=RAMP,title=f"{marker}: {outcome_label} by quartile")
            chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
            p_text = "Source analysis: NLR/WBC showed stronger early-mortality signal than hs-CRP."
            finding = f"Across quartiles, the observed {outcome_label.lower()} rate changes with {marker.lower()}. The chart lets you see whether the highest-marker group separates from the lowest group in this dataset."
        interpretation = "The combined inflammation + albumin analysis showed the clearest early mortality separation when both conditions were present. NLR was also practical because it is derived from routine blood-count components and was available for most patients."
        talk_track = f"I selected Inflammation & Nutrition, then {marker}, then {outcome_label}. I am not saying inflammation causes death; I am showing how the observed outcome rate differs across the groups created in our analysis."

    # ---------------- BNP / cardiac biomarkers ----------------
    elif area == "❤️ Cardiac Biomarkers":
        marker = st.selectbox("2. Select Cardiac Marker", ["BNP", "Troponin"])
        if marker == "BNP":
            outcome_label = st.selectbox("3. Select Outcome", ["6-Month Mortality", "6-Month Readmission", "28-Day Mortality"])
            q=insight_df.dropna(subset=["brain_natriuretic_peptide"]).copy()
            q["BNP group"]="Capped 5000"
            below=q["brain_natriuretic_peptide"]<5000
            q.loc[below,"BNP group"]=pd.qcut(q.loc[below,"brain_natriuretic_peptide"],4,labels=["Q1 (lowest)","Q2","Q3","Q4"]).astype(str)
            target=outcome_series(outcome_label)
            table=q.groupby("BNP group",observed=True)[target].mean().mul(100).reindex(["Q1 (lowest)","Q2","Q3","Q4","Capped 5000"]).reset_index()
            table.columns=["BNP group","Outcome rate (%)"]
            chart=px.bar(table,x="BNP group",y="Outcome rate (%)",text_auto=".1f",color="BNP group",color_discrete_sequence=RAMP,title=f"BNP groups: {outcome_label}")
            chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
            if outcome_label=="6-Month Mortality":
                p_text="BNP ≥ 708 vs < 708: p < 0.001"
                finding="Six-month mortality increased across the higher BNP groups, reaching about <b>6.0%</b> in the capped 5000 group versus about <b>1.3%</b> in the two lowest groups."
            elif outcome_label=="6-Month Readmission":
                p_text="BNP ≥ 708 comparison: source analysis reported p = 0.24 for 6-month readmission"
                finding="Readmission showed a weaker pattern than mortality: the rates varied across BNP groups but the source analysis did not find a statistically clear 6-month readmission association."
            else:
                p_text="Descriptive comparison"
                finding="Higher BNP groups showed higher observed 28-day mortality in the source analysis."
        else:
            outcome_label=st.selectbox("3. Select Outcome",["28-Day Mortality","6-Month Mortality"])
            target=outcome_series(outcome_label)
            col="troponin_i" if "troponin_i" in insight_df.columns else next((c for c in insight_df.columns if "troponin" in c.lower()),None)
            if col is None:
                table=pd.DataFrame({"Status":["Troponin column not available in current dashboard data"],"Outcome rate (%)":[0]})
                chart=px.bar(table,x="Status",y="Outcome rate (%)",title="Troponin")
                finding="The current cleaned file does not expose a troponin column with a matching name, so this marker cannot be displayed safely."
                p_text="Not available"
            else:
                q=insight_df.dropna(subset=[col,target]).copy(); q["Group"]=np.where(q[col]>q[col].median(),"Above median","At/below median")
                table=q.groupby("Group")[target].mean().mul(100).reindex(["At/below median","Above median"]).reset_index(); table.columns=["Group","Outcome rate (%)"]
                chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".1f",color="Group",color_discrete_sequence=["#9FB7BE",DEATH],title=f"Troponin: {outcome_label}")
                chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
                finding=f"Patients above the median troponin level had an observed {outcome_label.lower()} rate of <b>{table.iloc[1,1]:.1f}%</b> versus <b>{table.iloc[0,1]:.1f}%</b> at or below the median."
                p_text="Descriptive comparison"
        interpretation="BNP showed a clearer relationship with mortality than with readmission in the source analysis. Biomarkers should be interpreted together with clinical severity and other patient characteristics."
        talk_track=f"I selected Cardiac Biomarkers, then {marker}, then {outcome_label}. The chart shows how the observed outcome changes across biomarker groups; this is an association from our dataset, not a standalone decision rule."

    # ---------------- Current severity ----------------
    elif area == "🩺 Current Clinical Severity":
        marker = st.selectbox("2. Select Severity Measure", ["Killip Grade", "NYHA Class"])
        outcome_label = st.selectbox("3. Select Outcome", ["In-Hospital Mortality", "28-Day Mortality", "6-Month Mortality"])
        target=outcome_series(outcome_label)
        col="killip_grade" if marker=="Killip Grade" else "nyha_cardiac_function_classification"
        q=insight_df.dropna(subset=[col,target]).copy()
        table=q.groupby(col)[target].mean().mul(100).reset_index(); table.columns=["Grade","Outcome rate (%)"]
        table["Grade"]=table["Grade"].apply(lambda x:f"Killip {int(x)}" if marker=="Killip Grade" else f"NYHA {int(x)}")
        chart=px.bar(table,x="Grade",y="Outcome rate (%)",text_auto=".1f",color="Outcome rate (%)",color_continuous_scale=["#EAF5F8",DEATH],title=f"{marker}: {outcome_label}")
        chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
        low=float(table.iloc[0,1]); high=float(table.iloc[-1,1])
        finding=f"Observed {outcome_label.lower()} increased across the severity scale in this dataset, from <b>{low:.1f}%</b> in the lowest observed group to <b>{high:.1f}%</b> in the highest observed group."
        p_text="Descriptive severity gradient"
        interpretation="Current clinical severity measures describe how sick the patient is at admission. In the source analysis, Killip grade showed a particularly strong mortality gradient, making current severity an important part of risk review."
        talk_track=f"I selected Current Clinical Severity, then {marker}, then {outcome_label}. The important point is the gradient: as the observed severity category increases, the outcome rate also changes. This is why the dashboard treats current severity as a major clinical signal."

    # ---------------- Past vs current ----------------
    elif area == "🧭 Current Severity vs Prior History":
        marker = st.selectbox("2. Select Comparison", ["Prior Cardiac History", "Current Killip Grade"])
        outcome_label = st.selectbox("3. Select Outcome", ["In-Hospital Mortality", "28-Day Mortality"])
        target=outcome_series(outcome_label)
        if marker=="Current Killip Grade":
            q=insight_df.dropna(subset=["killip_grade",target]); table=q.groupby("killip_grade")[target].mean().mul(100).reset_index(); table.columns=["Group","Outcome rate (%)"]; table["Group"]=table["Group"].apply(lambda x:f"Killip {int(x)}")
            chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".1f",color="Outcome rate (%)",color_continuous_scale=["#EAF5F8",DEATH],title=f"Current severity: {outcome_label}")
            chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
            finding=f"Current Killip severity showed a much wider observed mortality range than prior-history indicators: the highest Killip group had <b>{table.iloc[-1,1]:.1f}%</b> {outcome_label.lower()} compared with <b>{table.iloc[0,1]:.1f}%</b> in the lowest group."
            p_text="Source model: severity ROC-AUC 0.87 for in-hospital death"
        else:
            histories={"Prior myocardial infarction":"myocardial_infarction","Prior heart failure":"congestive_heart_failure","Peripheral vascular disease":"peripheral_vascular_disease"}
            h=st.selectbox("History item",list(histories.keys()))
            col=histories[h]
            q=insight_df.dropna(subset=[col,target]); table=q.groupby(col)[target].mean().mul(100).reset_index(); table["Group"]=table[col].map({0:"No history",1:"History present"}); table=table[["Group",target]].rename(columns={target:"Outcome rate (%)"})
            chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".1f",color="Group",color_discrete_sequence=["#9FB7BE",TEAL2],title=f"{h}: {outcome_label}")
            chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
            finding=f"The observed difference associated with {h.lower()} is relatively small compared with the much larger gradient seen across current severity levels in the source analysis."
            p_text="Source model: history-only ROC-AUC 0.49; severity-only ROC-AUC 0.87 for in-hospital death"
        interpretation="The project analysis suggests that current clinical severity carries more discriminating information for early mortality than the selected prior-history indicators alone. Past history still provides context, but it should not be treated as a substitute for the patient's current presentation."
        talk_track="This is the 'now versus past' analysis. I use it to explain that a diagnosis in the history section and the patient's current severity are different kinds of information. In our analysis, current severity separated mortality outcomes much more strongly."

    # ---------------- Blood gas ----------------
    elif area == "🫁 Blood Gas":
        marker = st.selectbox("2. Select Blood-Gas Marker", ["Lactate", "pH", "Oxygen Saturation"])
        outcome_label = st.selectbox("3. Select Outcome", ["In-Hospital Mortality"])
        target=outcome_series(outcome_label)
        configs={
            "Lactate":("lactate",lambda x:x>2.2,"Lactate > 2.2","Lactate ≤ 2.2","p = 0.006"),
            "pH":("ph",lambda x:x<7.35,"pH < 7.35","pH ≥ 7.35","p = 0.562"),
            "Oxygen Saturation":("oxygen_saturation",lambda x:x<93,"O2 saturation < 93%","O2 saturation ≥ 93%","p = 0.575")}
        col,fn,lab1,lab0,p_text=configs[marker]
        q=insight_df.dropna(subset=[col,target]).copy(); q["Group"]=np.where(fn(q[col]),lab1,lab0)
        table=q.groupby("Group")[target].mean().mul(100).reindex([lab0,lab1]).reset_index(); table.columns=["Group","Outcome rate (%)"]
        chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".2f",color="Group",color_discrete_sequence=["#9FB7BE",DEATH],title=f"{marker}: {outcome_label}")
        chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
        finding=f"The source analysis found the clearest association for elevated lactate: the observed in-hospital death rate was <b>{table.iloc[1,1]:.2f}%</b> versus <b>{table.iloc[0,1]:.2f}%</b>, with <b>{p_text}</b>."
        if marker != "Lactate":
            finding=f"For {marker.lower()}, the observed difference in in-hospital mortality was small and the source analysis did not show a statistically clear association (<b>{p_text}</b>)."
        interpretation="Blood-gas measures do not all behave the same way. Lactate showed the clearest signal in the source analysis, while pH and oxygen saturation did not show statistically clear differences at the selected thresholds."
        talk_track=f"I selected Blood Gas, then {marker}. This is useful because it shows that not every abnormal-looking marker automatically carries the same outcome signal in our dataset. Lactate stood out more clearly than the other two measures."

    # ---------------- Anemia ----------------
    elif area == "🩸 Anemia":
        marker = st.selectbox("2. Select Hemoglobin View", ["Anemia Severity"])
        outcome_label = st.selectbox("3. Select Outcome", ["6-Month Mortality", "6-Month Readmission"])
        target=outcome_series(outcome_label)
        q=insight_df.dropna(subset=["anemia_level",target]).copy(); table=q.groupby("anemia_level",observed=True)[target].mean().mul(100).reset_index(); table.columns=["Anemia level","Outcome rate (%)"]
        chart=px.bar(table,x="Anemia level",y="Outcome rate (%)",text_auto=".1f",color="Anemia level",color_discrete_sequence=RAMP,title=f"Anemia severity: {outcome_label}")
        chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
        finding=f"The observed {outcome_label.lower()} rate varies across anemia severity groups. In the source analysis, severe anemia showed the clearest mortality difference, while mild and moderate anemia were common but less separated."
        p_text="Source analysis: severe anemia was associated with higher 6-month mortality"
        interpretation="Anemia is common in the dataset, but the most notable mortality signal was concentrated in severe anemia. This supports treating anemia as one component of the broader clinical picture rather than as a standalone explanation."
        talk_track=f"I selected Anemia and {outcome_label}. The key point is not that every degree of anemia has the same effect. The source analysis found the clearest mortality difference in the severe group."

    # ---------------- Readmission ----------------
    else:
        marker = st.selectbox("2. Select Readmission Factor", ["NYHA Class", "Killip Grade", "CKD Stage"])
        outcome_label = st.selectbox("3. Select Outcome", ["6-Month Readmission", "6-Month Mortality"])
        target=outcome_series(outcome_label)
        if marker=="NYHA Class": col="nyha_cardiac_function_classification"; prefix="NYHA"
        elif marker=="Killip Grade": col="killip_grade"; prefix="Killip"
        else: col="ckd_stage"; prefix="CKD"
        q=insight_df.dropna(subset=[col,target]).copy()
        table=q.groupby(col,observed=True)[target].mean().mul(100).reset_index(); table.columns=["Group","Outcome rate (%)"]
        table["Group"]=table["Group"].apply(lambda x:f"{prefix} {x}" if prefix!="CKD" else str(x))
        chart=px.bar(table,x="Group",y="Outcome rate (%)",text_auto=".1f",color="Outcome rate (%)",color_continuous_scale=["#EAF5F8",READMIT if "Readmission" in outcome_label else DEATH],title=f"{marker}: {outcome_label}")
        chart.update_layout(showlegend=False,yaxis_title="% of patients",xaxis_title="")
        finding=f"The observed {outcome_label.lower()} rate changes across {marker.lower()} categories. This view is intended to show the pattern across groups rather than claim that the factor alone determines an individual patient's outcome."
        p_text="Descriptive comparison"
        interpretation="Readmission is a different outcome from mortality and is influenced by clinical status as well as factors beyond the hospital record. The dashboard therefore presents readmission patterns separately."
        talk_track=f"I selected Readmission Patterns, then {marker}, then {outcome_label}. I use this to explain how the outcome varies across patient groups, while recognizing that readmission is influenced by more than clinical severity alone."

    # ---------------- Render selected insight ----------------
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        if chart is not None:
            st.plotly_chart(style(chart, 430), width="stretch")
    with right:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge'>{area.replace('🫘 ','').replace('🩸 ','').replace('❤️ ','').replace('🩺 ','').replace('🧭 ','').replace('🫁 ','').replace('🔁 ','')}</span>", unsafe_allow_html=True)
        st.markdown("### 🔎 Finding")
        st.markdown(f"{finding}", unsafe_allow_html=True)
        if p_text:
            st.markdown(f"<p><b>Evidence:</b> {p_text}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='found'><b>What this means:</b> {interpretation}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🗣️ How to explain this in your presentation"):
        st.write(talk_track)
        st.caption("Tip: explain the selected group, the observed outcome difference, and what the statistical evidence says. Avoid describing an association as proof of causation.")

    st.markdown("""
    <div class='section'>
    <h3 style='color:#073B4C;margin-top:0'>⭐ Overall project takeaways</h3>
    <div class='found'><b>1. Current severity matters.</b> NYHA and especially Killip provide a direct view of how sick the patient is at admission, and the analysis shows clear mortality gradients across severity levels.</div>
    <div class='found'><b>2. Kidney function adds important context.</b> eGFR, creatinine and urea were associated with higher observed 6-month mortality/readmission, with creatinine showing the largest differences among the kidney markers tested.</div>
    <div class='found'><b>3. Biomarkers are outcome-specific.</b> BNP showed a clearer relationship with mortality than with readmission, while NLR showed a useful early-mortality signal.</div>
    <div class='found'><b>4. Readmission and mortality should be examined separately.</b> Returning to hospital is influenced by clinical and non-clinical factors, so the same marker does not necessarily behave the same way for both outcomes.</div>
    <div class='found'><b>5. The ANN belongs at the end of the story.</b> The predictive model combines multiple patient characteristics into an analytical risk estimate after the descriptive and clinical patterns have been understood.</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# 5. MODEL PERFORMANCE
# =====================================================================
elif page == "🤖 Model Performance":
    st.markdown("<div class='hdr'><h1>🤖 Model Performance</h1>"
                "<p>How well our models find high-risk patients, tested on patients they had never seen</p></div>",
                unsafe_allow_html=True)

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
                                       df[(df["outcome_during_hospitalization"] != "Dead") & (df["death_within_6_months"] == 0)].reset_index(drop=True))}
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

    st.subheader("Our key models at a glance")
    summary = pd.DataFrame({
        "What we predicted": ["Death within 28 days", "Death within 6 months", "Death within 28 days", "Came back within 6 months"],
        "Using": ["Bedside check only (Killip + NYHA)", "Bedside check + 6 routine blood tests",
                  "NLR from the routine blood count", "21 admission measures"],
        "ROC-AUC": ["0.85", "0.81", "0.70", "0.61"],
        "In simple words": ["Excellent with just a 30-second exam", "Best overall death model", "A free test with useful signal",
                            "Weak, but top-risk group returns 2x as often"]})
    st.dataframe(summary, hide_index=True, width="stretch")

    # ---------------- Patient risk check ----------------
    st.subheader("🩺 Try it: Patient Risk Check")
    st.caption("Uses the 6-month death model. It supports the doctor's judgement; it does not replace it.")

    @st.cache_resource
    def final_model():
        model = logistic().fit(df[DEATH_FEATURES].astype(float), df["death_within_6_months"])
        prob = cv_probs(df, DEATH_FEATURES, "death_within_6_months", "Logistic Regression", repeats=3)
        edges = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
        rate = pd.Series(df["death_within_6_months"].values).groupby(np.digitize(prob, edges)).mean() * 100
        return model, edges, rate

    model, edges, death_rate = final_model()
    with st.form("patient"):
        c1, c2, c3, c4 = st.columns(4)
        nyha = c1.selectbox("NYHA class (symptoms)", [2, 3, 4], index=1)
        killip = c2.selectbox("Killip grade (fluid / shock)", [1, 2, 3, 4], index=1)
        bnp = c3.number_input("BNP (pg/mL)", 10.0, 5000.0, 750.0)
        trop = c4.number_input("Troponin (pg/mL)", 0.0, 50000.0, 55.0)
        c5, c6, c7, c8 = st.columns(4)
        neut = c5.number_input("Neutrophils (x10^9/L)", 0.1, 50.0, 5.0)
        lymph = c6.number_input("Lymphocytes (x10^9/L)", 0.05, 20.0, 1.0)
        alb = c7.number_input("Albumin (g/L)", 10.0, 60.0, 37.0)
        hbv = c8.number_input("Hemoglobin (g/L)", 30.0, 200.0, 115.0)
        c9, c10, c11, _ = st.columns(4)
        na = c9.number_input("Sodium (mmol/L)", 110.0, 160.0, 139.0)
        egfr = c10.number_input("eGFR (kidney)", 1.0, 200.0, 60.0)
        sbp_in = c11.number_input("Systolic BP (mmHg)", 50.0, 250.0, 130.0)
        submitted = st.form_submit_button("Check risk", type="primary")

    if submitted:
        nlr_val = neut / lymph
        x = pd.DataFrame([[nyha, killip, np.log1p(bnp), np.log1p(trop), np.log(nlr_val), alb, hbv, na]], columns=DEATH_FEATURES)
        grp = int(np.digitize(model.predict_proba(x)[0, 1], edges))
        names = ["Lowest", "Low", "Middle", "High", "Highest"]
        colours = [RAMP[0], RAMP[1], "#F2C14E", "#E07A5F", ALERT]
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown(f"### Risk group: <span style='color:{colours[grp]}'>{names[grp]}</span>", unsafe_allow_html=True)
            kpi("⚠️", "Similar patients who died within 6 months", f"{death_rate.iloc[grp]:.1f}%")
            flags = [("Fluid in lungs or shock (Killip 3-4)", killip >= 3), ("Symptoms at rest (NYHA IV)", nyha == 4),
                     ("Low blood pressure (below 90)", sbp_in < 90), ("Weak kidneys (eGFR below 45)", egfr < 45),
                     ("Severe anemia (hemoglobin below 80)", hbv < 80), (f"High NLR ({nlr_val:.1f})", nlr_val >= 8.7)]
            shown = [n for n, on in flags if on]
            st.write("")
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
    st.markdown("<div class='hdr'><h1>📌 Key Takeaways & Conclusion</h1><p>What the HeartFailure analysis tells us</p></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class='section'>
        <h4 style='color:#073B4C;margin-top:0'>⭐ Key Takeaways</h4>
        <ul>
        <li><b>The dataset supports a complete hospital-episode view:</b> demographic, cardiac, laboratory, history, treatment and outcome information can be examined together at the patient level.</li>
        <li><b>Current severity is important:</b> NYHA and Killip provide admission-level measures that can be compared with mortality and readmission outcomes.</li>
        <li><b>Multiple organ systems matter:</b> kidney function, blood markers, inflammation, nutrition and cardiac biomarkers provide complementary signals rather than a single explanation.</li>
        <li><b>Readmission and mortality are different outcomes:</b> a patient may have a higher observed likelihood of returning without having the same mortality pattern, so they should be analyzed separately.</li>
        <li><b>Feature engineering improves interpretation:</b> clinically meaningful groups and warning flags make complex laboratory and clinical values easier to explore.</li>
        <li><b>Machine learning adds a patient-level risk view:</b> Logistic Regression, Random Forest and ANN can be compared using held-out/cross-validated predictions and multiple performance metrics.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class='section'>
        <h4 style='color:#073B4C;margin-top:0'>🏥 How the dashboard can be used</h4>
        <ul>
        <li><b>At admission:</b> review current clinical severity together with prior history, comorbidities and baseline laboratory results.</li>
        <li><b>During analysis:</b> use the descriptive views to understand which patient groups and biomarkers are associated with different outcomes.</li>
        <li><b>For risk review:</b> use model probabilities as an analytical flag for closer review, not as a diagnosis or automatic treatment decision.</li>
        <li><b>For follow-up planning:</b> examine readmission patterns separately from mortality because they represent different patient outcomes.</li>
        <li><b>For quality improvement:</b> compare observed patterns across patient groups and identify areas that may deserve further clinical investigation.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section'>
    <h4 style='color:#073B4C;margin-top:0'>🏁 Conclusion</h4>
    <p>
    The <b>HeartFailure Clinical Explorer</b> brings the project workflow into one place: the source hospital records are integrated and cleaned, clinically meaningful features are created, descriptive and outcome analyses are performed, and machine-learning models are evaluated for mortality-risk prediction.
    </p>
    <p>
    The main value of the dashboard is not a single number or model. It is the ability to connect <b>patient characteristics → current clinical severity → laboratory and cardiac signals → observed outcomes → model-based risk estimates</b> in a form that can be explored patient by patient or across the population.
    </p>
    <p>
    The results should be interpreted as <b>associations and research findings</b>. The source dataset comes from a single hospital and is retrospective, and the original documentation notes that models developed from it may not generalize to other settings. Therefore, this dashboard is intended for <b>education, analytics and research</b>, not for autonomous diagnosis or treatment decisions.
    </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How the HeartFailure project was built"):
        st.markdown("""
        **Data source:** PhysioNet HeartFailure dataset, version 1.3.

        **Workflow:** seven source tables → patient-level integrated dataset → data cleaning → feature engineering → descriptive analysis → clinical outcome analysis → predictive modeling → patient-level exploration.

        **Models:** Logistic Regression, Random Forest and Artificial Neural Network (ANN). Performance is reviewed with multiple metrics rather than a single accuracy value.

        **Important limitation:** this is a retrospective single-center dataset. Model results are not externally validated and should not be interpreted as proof of causation or as a clinical decision rule.
        """)
