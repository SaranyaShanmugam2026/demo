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
                                   "📊 Insights", "🤖 Model Performance", "📌 Key Takeaways & Conclusion"],
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
elif page == "📊 Insights":
    st.markdown("<div class='hdr'><h1>📊 Insights</h1><p>Descriptive • Prescriptive • Predictive, in simple words</p></div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("🔁", "Came back within 6 months", pct(df["re_admission_within_6_months"].mean()))
    with c2: kpi("⚠️", "Died within 6 months", pct(df["death_within_6_months"].mean()))
    with c3: kpi("❤️", "Severe symptoms (NYHA III–IV)", pct((df["nyha_cardiac_function_classification"] >= 3).mean()))
    with c4: kpi("🧪", "Median BNP (heart strain)", f"{df['brain_natriuretic_peptide'].median():.0f} pg/mL")
    st.write("")

    d28 = "death_within_28_days"
    tabs = st.tabs(["👥 Patients", "💊 Medicines", "🫘 Kidneys", "🩸 Anemia", "🩺 Blood Pressure",
                    "🩺 Current Clinical Severity", "🧭 Current Severity vs Prior History", "🧪 Blood Test", "🔁 Readmission Patterns"])

    # ---------- Descriptive ----------
    with tabs[0]:
        badge("Descriptive")
        left, right = st.columns(2)
        with left:
            m = pd.Series({
                "High BNP (heart under strain)": df["bnp_elevated_flag"].mean(),
                "Heart muscle damage": df["troponin_elevated_flag"].mean(),
                "Anemia": df["anemia_level"].isin(["Mild", "Moderate", "Severe"]).sum() / df["anemia_level"].notna().sum(),
                "Weak kidneys (eGFR < 60)": (df["glomerular_filtration_rate"] < 60).sum() / df["glomerular_filtration_rate"].notna().sum(),
                "Underweight": (df["bmi_category"] == "Underweight").mean(),
                "Diabetes": df["diabetes"].mean(),
            }).sort_values() * 100
            fig = px.bar(m, orientation="h", text_auto=".0f", color_discrete_sequence=[TEAL2], title="How common each problem is (%)")
            fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="")
            st.plotly_chart(style(fig), width="stretch")
        with right:
            sev = pd.crosstab(df["nyha_cardiac_function_classification"], df["killip_grade"])
            sev.index = [f"NYHA {i}" for i in sev.index]
            sev.columns = [f"Killip {c}" for c in sev.columns]
            fig = px.imshow(sev, text_auto=True, color_continuous_scale=["#EAF5F8", TEAL2, NAVY],
                            title="Patients by symptom level (NYHA) and fluid/shock level (Killip)")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style(fig), width="stretch")
        found("Almost every patient has a strained heart (92% high BNP) and most show heart muscle damage (84%). "
              "Anemia (61%) is more common than diabetes (23%). 1 in 4 patients is underweight, a sign of frailty.")
        todo("This is a very sick, frail group, so extra checks at admission are worth the effort.")

    with tabs[1]:
        badge("Descriptive")
        drugs = pd.Series({
            "Water tablets (diuretic)": ((df["Furosemide injection"] + df["Furosemide tablet"] + df["Torasemide tablet"] +
                                          df["Hydrochlorothiazide tablet"]) > 0).mean(),
            "Spironolactone": df["Spironolactone tablet"].mean(),
            "ACE inhibitor / ARB": ((df["Benazepril hydrochloride tablet"] + df["Valsartan Dispersible tablet"]) > 0).mean(),
            "Beta-blocker": ((df["Metoprolol Succinate Sustained-release tablet"] + df["metoprolol tartrate injection"]) > 0).mean(),
        }).sort_values() * 100
        fig = px.bar(drugs, orientation="h", text_auto=".0f", color_discrete_sequence=[GREEN], title="Recommended heart medicines given (%)")
        fig.update_layout(showlegend=False, xaxis_title="% of patients", yaxis_title="")
        st.plotly_chart(style(fig, 340), width="stretch")
        found("Almost everyone gets water tablets (96%), but <b>only about 4 in 10</b> get the key long-term heart medicines "
              "(ACE inhibitor/ARB, beta-blocker). Only 19% get the full recommended combination. The average patient takes 8 medicines.")
        todo("Check before discharge that every suitable patient is on the recommended long-term medicines.")

    # ---------- Prescriptive ----------
    with tabs[2]:
        badge("Prescriptive")
        k = df.dropna(subset=["ckd_stage"])
        g = k.groupby("ckd_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "As kidneys get weaker, more patients die (kidney stage, worse →)"), width="stretch")
        found("Weak kidneys and a weak heart pull each other down. Deaths rise from <b>1.6%</b> with healthy kidneys to "
              "<b>9.2%</b> with kidney failure, and returns peak at <b>50%</b> in stage G3b.")
        todo("Treat patients with eGFR below 45 as high risk: check potassium, dose water tablets carefully, "
             "and see them again within 2 weeks of going home.")

    with tabs[3]:
        badge("Prescriptive")
        a = df.dropna(subset=["anemia_level"])
        g = a.groupby("anemia_level", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Only severe anemia stands out"), width="stretch")
        found("Mild and moderate anemia are very common but add little risk. <b>Severe anemia (hemoglobin below 80)</b> "
              "nearly <b>triples</b> the 6-month death rate (6.8% vs about 2.5%).")
        todo("Severe anemia is associated with higher observed mortality in this dataset; this can be treated as a review flag alongside the broader clinical picture.")

    with tabs[4]:
        badge("Prescriptive")
        b = df.dropna(subset=["bp_stage"])
        g = b.groupby("bp_stage", observed=True)
        table = pd.DataFrame({"Readmitted in 6 months": g["re_admission_within_6_months"].mean() * 100,
                              "Died in 6 months": g["death_within_6_months"].mean() * 100})
        st.plotly_chart(two_outcomes(table, "Low blood pressure is rare but dangerous"), width="stretch")
        found("Only 20 patients arrived with low blood pressure (below 90), but <b>9 in 10 had symptoms at rest</b> and "
              "<b>8 in 10 had fluid in the lungs or shock</b>. They did <b>not</b> get heart-support drips more often. "
              "Patients with higher blood pressure came back <b>less</b> often, because their heart still has pumping strength.")
        todo("Very low admission blood pressure is associated with higher observed risk in this dataset and can be highlighted for clinical review alongside other severity measures.")

    # ---------- Predictive ----------
    with tabs[5]:
        badge("Predictive")
        left, right = st.columns(2)
        with left:
            kil = df.groupby("killip_grade")[d28].mean() * 100
            st.plotly_chart(bar([f"Killip {x}" for x in kil.index], kil.values, "Deaths within 28 days by Killip grade (%)", RAMP[1:]),
                            width="stretch")
        with right:
            nyha4 = np.where(df["nyha_cardiac_function_classification"] == 4, "Symptoms at rest", "Symptoms on activity")
            k34 = np.where(df["killip_grade"] >= 3, "Fluid in lungs / shock", "No / mild fluid")
            grid = (pd.crosstab(nyha4, k34, values=df[d28], aggfunc="mean") * 100).round(1)
            grid = grid.loc[["Symptoms on activity", "Symptoms at rest"], ["No / mild fluid", "Fluid in lungs / shock"]]
            fig = px.imshow(grid, text_auto=True, color_continuous_scale=["#EAF5F8", DEATH],
                            title="Deaths within 28 days (%): two bedside scores together")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(style(fig), width="stretch")
        found("A 30-second bedside exam (Killip grade) sorts patients very well. <b>None of 527 Killip 1 patients died</b> "
              "within 28 days, while <b>1 in 4 Killip 4 patients died</b>. Adding symptom level makes it sharper: "
              "0.4% vs <b>11.9%</b> deaths, a <b>30 times</b> difference.")
        todo("The observed mortality gradient across Killip grades supports using current severity measures as an important part of risk review. It should not be interpreted as a standalone disposition rule.")

    with tabs[6]:
        badge("Predictive")
        hist = pd.Series({
            "Old heart attack": df.loc[df["myocardial_infarction"] == 1, d28].mean(),
            "No old heart attack": df.loc[df["myocardial_infarction"] == 0, d28].mean(),
            "Past heart failure": df.loc[df["congestive_heart_failure"] == 1, d28].mean(),
            "No past heart failure": df.loc[df["congestive_heart_failure"] == 0, d28].mean(),
        }) * 100
        left, right = st.columns(2)
        with left:
            st.plotly_chart(bar(list(hist.index), hist.values, "Prior history: observed 28-day mortality", ["#9FB7BE"] * 4),
                            width="stretch")
        with right:
            kil = df.groupby("killip_grade")[d28].mean() * 100
            st.plotly_chart(bar([f"Killip {x}" for x in kil.index], kil.values, "Current clinical severity: observed 28-day mortality", RAMP[1:]),
                            width="stretch")
        found("A patient's <b>past</b> (old heart attack, earlier heart failure) tells us almost nothing about who will die: "
              "about 2% either way. How sick the patient is <b>today</b> tells us almost everything.")
        todo("Use current severity and prior history together as complementary information during clinical review; this analysis describes associations and is not a standalone care decision rule.")

    with tabs[7]:
        badge("Predictive")
        left, right = st.columns(2)
        with left:
            q = pd.qcut(df["nlr"], 4, labels=["Lowest NLR", "Low", "High", "Highest NLR"])
            nq = df.groupby(q, observed=True)[d28].mean() * 100
            st.plotly_chart(bar(list(nq.index.astype(str)), nq.values, "Deaths within 28 days by NLR level (%)", RAMP[1:]),
                            width="stretch")
        with right:
            avail = pd.Series({"NLR (routine blood count)": df["nlr"].notna().mean() * 100,
                               "hs-CRP (special test)": df["hs_crp"].notna().mean() * 100})
            st.plotly_chart(bar(list(avail.index), avail.values, "How many patients had the test (%)", [GREEN, "#9FB7BE"]),
                            width="stretch")
        found("NLR comes free with the routine blood count. The highest NLR group had <b>8 times</b> the early death rate "
              "of the lowest (3.4% vs 0.4%). The special inflammation test (hs-CRP) did not help, because "
              "<b>more than half of patients were never tested</b>.")
        todo("Calculate NLR for every patient and flag NLR of 8.7 or more.")

    with tabs[8]:
        badge("Predictive")
        alive = df[(df["outcome_during_hospitalization"] != "Dead") & (df["death_within_6_months"] == 0)].reset_index(drop=True)
        with st.spinner("Scoring patients..."):
            prob = cv_probs(alive, READMIT_FEATURES, "re_admission_within_6_months", "Logistic Regression", repeats=3)
        groups = pd.qcut(prob, 5, labels=["Lowest risk", "Low", "Middle", "High", "Highest risk"])
        by_g = alive.groupby(groups, observed=True)["re_admission_within_6_months"].mean() * 100
        st.plotly_chart(bar(list(by_g.index.astype(str)), by_g.values,
                            "Patients who actually came back, by predicted risk group (%)", RAMP), width="stretch")
        found(f"Coming back is harder to predict than death, because it also depends on life outside the hospital "
              f"(home support, taking medicines). Still, our model's <b>highest-risk group came back about twice as often</b> "
              f"({by_g.iloc[-1]:.0f}%) as the lowest-risk group ({by_g.iloc[0]:.0f}%). Main drivers: severe symptoms, "
              f"weak kidneys and other diseases.")
        todo("Give the highest-risk group a follow-up phone call and an early clinic visit after discharge.")


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
