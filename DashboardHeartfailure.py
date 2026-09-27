import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HeartCare AI | Heart Failure Clinical Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.stApp { background:#F4F9FB; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#073B4C 0%,#0B5D6B 55%,#087F5B 100%); }
section[data-testid="stSidebar"] * { color:white !important; }
.hospital-header { background:linear-gradient(90deg,#073B4C,#087F5B); padding:26px 30px; border-radius:16px; color:white; margin-bottom:22px; box-shadow:0 5px 16px rgba(0,0,0,.08); }
.hospital-header h1 { margin:0; font-size:31px; }
.hospital-header p { margin:7px 0 0; opacity:.92; }
.section-card { background:white; padding:20px; border-radius:15px; box-shadow:0 3px 12px rgba(0,0,0,.05); margin-bottom:18px; }
.metric-card { background:white; padding:18px; border-radius:15px; border-left:5px solid #087F5B; box-shadow:0 3px 12px rgba(0,0,0,.06); min-height:112px; }
.metric-icon { font-size:27px; }
.metric-title { color:#637B83; font-size:14px; font-weight:600; }
.metric-value { color:#073B4C; font-size:27px; font-weight:700; margin-top:5px; }
.info-box { background:#EAF5F8; border-left:5px solid #087F9B; padding:16px; border-radius:9px; }
.takeaway { background:#F0F8F4; border-left:5px solid #087F5B; padding:16px; border-radius:9px; margin:8px 0; }
.warning-box { background:#FFF8E8; border-left:5px solid #D89B00; padding:16px; border-radius:9px; }
.small-note { color:#60757D; font-size:13px; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD THE CLEANED EXCEL DIRECTLY FROM THE GITHUB REPOSITORY
# ============================================================
DATA_FILE = Path(__file__).parent / "Cardiac_Cleaned_Data.xlsb"

@st.cache_data
def _load_data():
    return pd.read_excel(DATA_FILE, engine="pyxlsb")

try:
    df = _load_data()
except Exception as e:
    st.error(f"Could not load Cardiac_Cleaned_Data.xlsb from the GitHub repository: {e}")
    st.stop()

# ============================================================
# HELPERS
# ============================================================
def find_col(names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n and str(n).strip().lower() in lookup:
            return lookup[str(n).strip().lower()]
    return None

def numeric_col(names):
    c = find_col(names)
    if c is None:
        return None
    s = pd.to_numeric(df[c], errors="coerce")
    return c if s.notna().sum() > 0 else None

def metric_card(icon, title, value):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-icon">{icon}</div>
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "N/A"

def outcome_rate(data, col):
    if col is None or len(data) == 0:
        return np.nan
    s = pd.to_numeric(data[col], errors="coerce")
    return s.mean() * 100

def add_binary_from_text(frame, source, positive_words):
    if source is None:
        return None
    text = frame[source].astype(str).str.lower().str.strip()
    new = f"__{source}_binary"
    frame[new] = text.isin(positive_words).astype(int)
    return new

# ============================================================
# ACTUAL COLUMN MAP
# ============================================================
age_col = numeric_col(["age", "age_years"])
agecat_col = find_col(["agecat", "age_category"])
gender_col = find_col(["gender", "sex"])
bmi_col = numeric_col(["bmi"])
weight_col = numeric_col(["weight"])
height_col = numeric_col(["height"])
nyha_col = find_col(["nyha_cardiac_function_classification", "nyha_cardiac", "nyha", "nyha_class"])
killip_col = find_col(["killip_grade", "killip", "killip_class"])
hf_type_col = find_col(["type_of_heart_failure", "heart_failure_type"])
lvedd_col = numeric_col(["lvedd_mm", "left_ventricular_end_diastolic_diameter_lv"])
ea_col = numeric_col(["ea", "e_a", "mitral_e_a"])
tricuspid_pressure_col = numeric_col(["tricuspid_valve_return_pressure", "tricuspid_pressure"])

outcome_text_col = find_col(["outcome_during_hospitalization"])
if outcome_text_col:
    df["__in_hospital_death"] = df[outcome_text_col].astype(str).str.lower().str.strip().eq("dead").astype(int)
    in_hospital_col = "__in_hospital_death"
else:
    in_hospital_col = find_col(["in_hospital_mortality", "hospital_mortality", "mortality"])

death28_col = find_col(["death_within_28_days", "mortality_28d", "mortality_28_day", "28_day_mortality"])
death3m_col = find_col(["death_within_3_months", "mortality_3m", "3_month_mortality"])
death6m_col = find_col(["death_within_6_months", "mortality_6m", "6_month_mortality"])
readm28_col = find_col(["re_admission_within_28_days", "readmission_within_28_days"])
readm3m_col = find_col(["re_admission_within_3_months", "readmission_within_3_months"])
readm6m_col = find_col(["re_admission_within_6_months", "readmission_within_6_months"])
ed6m_col = find_col(["return_to_emergency_department_within_6_months", "ed_return_6m"])
los_col = numeric_col(["dischargeday", "length_of_stay", "length_of_stay_days"])
readm_time_col = numeric_col(["readmission_time_days_from_admission"])
ed_time_col = numeric_col(["time_to_emergency_department_within_6_months"])

# Biomarkers / clinical domains
biomarker_aliases = {
    "BNP": ["brain_natriuretic_peptide", "bnp"],
    "High-sensitivity troponin": ["high_sensitivity_troponin", "high_sensitivity_troponin_i", "troponin_i", "troponin"],
    "hs-CRP": ["hs_crp", "hs-crp", "hscrp", "high_sensitivity_crp"],
    "WBC": ["wbc", "white_blood_cell_count"],
    "NLR": ["nlr", "neutrophil_lymphocyte_ratio"],
    "Albumin": ["albumin"],
    "Hemoglobin": ["hemoglobin", "hb"],
    "RDW": ["rdw"],
    "Creatinine": ["creatinine_enzymatic_method", "creatinine", "serum_creatinine"],
    "eGFR": ["egfr", "e_gfr"],
    "Urea": ["urea", "blood_urea_nitrogen", "bun"],
    "Cystatin": ["cystatin", "cystatin_c"],
    "D-dimer": ["d_dimer", "d-dimer", "ddimer"],
    "INR": ["inr"],
    "Sodium": ["sodium_ion", "sodium", "na"],
    "Potassium": ["potassium_ion", "potassium", "k"],
    "Lactate": ["lactate"],
    "pH": ["ph", "pH"],
    "Oxygen saturation": ["oxygen_saturation", "spo2", "oxygen_saturation_blood_gas"],
    "Total bilirubin": ["total_bilirubin"],
    "AST/ALT ratio": ["ast_alt_ratio", "ast_alt", "ast_alt_ratio_value"],
    "LVEF": ["lvef"],
    "LVEDD": ["lvedd_mm", "left_ventricular_end_diastolic_diameter_lv"],
    "CCI": ["cci_score", "charlson_comorbidity_index"],
    "Pulse": ["pulse"],
    "Systolic BP": ["systolic_blood_pressure", "sbp"],
    "Diastolic BP": ["diastolic_blood_pressure", "dbp"],
    "Respiration": ["respiration", "respiratory_rate"],
    "CO2": ["partial_pressure_of_carbon_dioxide", "co2", "total_carbon_dioxide"],
}
biomarkers = {label: find_col(aliases) for label, aliases in biomarker_aliases.items()}
biomarkers = {k:v for k,v in biomarkers.items() if v is not None}

# ============================================================
# FEATURE ENGINEERING — analysis-ready derived variables
# ============================================================
if bmi_col:
    bmi_num = pd.to_numeric(df[bmi_col], errors="coerce")
    df["BMI Category"] = pd.cut(
        bmi_num,
        bins=[0,18.5,25,30,np.inf],
        labels=["Underweight","Normal","Overweight","Obese"],
        right=False
    )
    df["Obesity Flag"] = (bmi_num >= 30).astype("Int64")

# Inflammation + albumin groups using the project's stated thresholds.
crp = biomarkers.get("hs-CRP")
wbc = biomarkers.get("WBC")
nlr = biomarkers.get("NLR")
alb = biomarkers.get("Albumin")
if any([crp,wbc,nlr,alb]):
    infl_parts = []
    if crp: infl_parts.append(pd.to_numeric(df[crp], errors="coerce") > 5)
    if wbc: infl_parts.append(pd.to_numeric(df[wbc], errors="coerce") > 10)
    if nlr: infl_parts.append(pd.to_numeric(df[nlr], errors="coerce") > 6)
    if infl_parts:
        df["Inflammation Flag"] = pd.concat(infl_parts, axis=1).any(axis=1).astype("Int64")
    if alb:
        df["Low Albumin Flag"] = (pd.to_numeric(df[alb], errors="coerce") < 35).astype("Int64")
    if "Inflammation Flag" in df and "Low Albumin Flag" in df:
        df["Inflammation + Albumin Group"] = np.select(
            [df["Inflammation Flag"].eq(0) & df["Low Albumin Flag"].eq(0),
             df["Inflammation Flag"].eq(1) & df["Low Albumin Flag"].eq(0),
             df["Inflammation Flag"].eq(0) & df["Low Albumin Flag"].eq(1),
             df["Inflammation Flag"].eq(1) & df["Low Albumin Flag"].eq(1)],
            ["Neither","Inflamed only","Low albumin only","Both"],
            default=np.nan
        )

# Cardiac + renal burden, median-based as used in the project analysis.
creat = biomarkers.get("Creatinine")
bnp = biomarkers.get("BNP")
if creat and bnp:
    c = pd.to_numeric(df[creat], errors="coerce")
    b = pd.to_numeric(df[bnp], errors="coerce")
    c_abn = c > c.median()
    b_abn = b > b.median()
    df["Cardiac-Renal Group"] = np.select(
        [~c_abn & ~b_abn, c_abn & ~b_abn, ~c_abn & b_abn, c_abn & b_abn],
        ["Neither abnormal","Kidney only","Cardiac only","Both abnormal"],
        default=np.nan
    )

# CCI bands.
cci = biomarkers.get("CCI")
if cci:
    cci_num = pd.to_numeric(df[cci], errors="coerce")
    df["CCI Band"] = pd.cut(cci_num, bins=[-np.inf,2,4,np.inf], labels=["0–2","3–4","5+"])

# Shock index.
sbp = biomarkers.get("Systolic BP")
pulse = biomarkers.get("Pulse")
if sbp and pulse:
    s = pd.to_numeric(df[sbp], errors="coerce")
    p = pd.to_numeric(df[pulse], errors="coerce")
    df["Shock Index"] = p / s.replace(0,np.nan)

# Medication burden: identify likely prescription-flag columns from common drug names.
drug_terms = [
    "aspirin","atorvastatin","furosemide","enoxaparin","digoxin","spironolactone",
    "warfarin","metoprolol","bisoprolol","carvedilol","ramipril","enalapril",
    "lisinopril","losartan","valsartan","sacubitril","clopidogrel","dobutamine",
    "milrinone","deslanoside","nitr","amiodarone","hydrochlorothiazide","dapagliflozin",
    "empagliflozin"
]
med_cols = []
for c in df.columns:
    cl = str(c).lower()
    if any(term in cl for term in drug_terms):
        med_cols.append(c)
if med_cols:
    df["Medication Burden"] = df[med_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px">
      <div style="font-size:48px">❤️</div>
      <h2>HeartCare AI</h2>
      <p style="font-size:13px">Heart Failure Clinical Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", [
        "🏠 Introduction",
        "📊 Overview",
        "🧹 Data Cleaning",
        "🧬 Feature Engineering",
        "📈 Descriptive Analysis",
        "💡 Prescriptive Analysis",
        "🤖 Predictive Analytics",
        "📊 Model Performance",
        "🔎 Patient Explorer",
        "⭐ Insights & Key Takeaways",
        "🏁 Conclusion",
    ])
    st.divider()
    st.caption(f"Data source: Cardiac_Cleaned_Data.xlsb")
    st.caption(f"Patients loaded: {len(df):,}")
    st.caption("For research and analytical use.")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hospital-header">
  <h1>❤️ Heart Failure Clinical Analytics</h1>
  <p>Data Cleaning • Feature Engineering • Descriptive • Prescriptive • Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INTRODUCTION
# ============================================================
if page == "🏠 Introduction":
    st.title("Project Introduction")
    st.markdown("""
    <div class="section-card">
    <h3>Research Question</h3>
    <p><b>Can demographic, clinical, cardiac, laboratory, nutritional, and responsiveness characteristics be combined in an Artificial Neural Network model to accurately predict in-hospital and 28-day mortality among patients with heart failure?</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("What this project does")
    cols = st.columns(4)
    with cols[0]: metric_card("🧹","Clean","7 source datasets")
    with cols[1]: metric_card("🔬","Describe","Patient + clinical patterns")
    with cols[2]: metric_card("💡","Prescribe","30 clinical questions")
    with cols[3]: metric_card("🤖","Predict","Mortality / readmission models")

    st.subheader("End-to-end workflow")
    st.markdown("""
    **Raw data → Data cleaning → Patient-level master dataset → Feature engineering → Descriptive analysis → Prescriptive clinical questions → Predictive modelling → Dashboard insights → Key takeaways → Conclusion**
    """)

    st.subheader("Why the dashboard exists")
    st.write("The dashboard is the presentation layer for the analysis. It should allow a hospital/research user to understand the cohort, inspect data quality, explore many clinical variables, review the 30 prescriptive questions, and inspect predictive-model performance without opening the notebooks.")

# ============================================================
# OVERVIEW
# ============================================================
elif page == "📊 Overview":
    st.title("Hospital Overview")
    st.caption("Patient population and overall outcomes from the cleaned patient-level dataset.")
    total = len(df)
    unique_ids = df["inpatient_number"].nunique() if "inpatient_number" in df.columns else total
    in_rate = outcome_rate(df, in_hospital_col)
    death28 = outcome_rate(df, death28_col)
    death6 = outcome_rate(df, death6m_col)
    readm6 = outcome_rate(df, readm6_col)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("👥","Patients",f"{total:,}")
    with c2: metric_card("🆔","Unique patients",f"{unique_ids:,}")
    with c3: metric_card("⚠️","In-hospital death",pct(in_rate))
    with c4: metric_card("📅","28-day death",pct(death28))
    with c5: metric_card("🔁","6-month readmission",pct(readm6))

    st.subheader("Outcome landscape")
    outcome_items = []
    for label,col in [("In-hospital death",in_hospital_col),("28-day death",death28_col),("3-month death",death3m_col),("6-month death",death6m_col),("28-day readmission",readm28_col),("3-month readmission",readm3m_col),("6-month readmission",readm6_col),("6-month ED return",ed6m_col)]:
        if col:
            outcome_items.append((label,outcome_rate(df,col)))
    if outcome_items:
        od = pd.DataFrame(outcome_items,columns=["Outcome","Rate"])
        st.plotly_chart(px.bar(od,x="Outcome",y="Rate",text=od["Rate"].map(lambda x:f"{x:.1f}%"),labels={"Rate":"Rate (%)"}),use_container_width=True)

    st.subheader("Cohort profile")
    c1,c2 = st.columns(2)
    with c1:
        if gender_col:
            g = df[gender_col].value_counts(dropna=False).reset_index()
            g.columns=["Gender","Patients"]
            st.plotly_chart(px.pie(g,names="Gender",values="Patients",hole=.45),use_container_width=True)
    with c2:
        if agecat_col:
            a = df[agecat_col].value_counts(dropna=False).reset_index()
            a.columns=["Age category","Patients"]
            st.plotly_chart(px.bar(a,x="Age category",y="Patients"),use_container_width=True)

# ============================================================
# DATA CLEANING
# ============================================================
elif page == "🧹 Data Cleaning":
    st.title("Data Cleaning & Quality")
    st.write("The dashboard uses the cleaned patient-level Excel file. The cleaning workflow was performed before dashboard analysis and was designed to preserve clinically meaningful information rather than blindly filling or deleting values.")

    st.subheader("Cleaning workflow used")
    steps = [
        ("1. Load seven source datasets","Demography, cardiac, hospitalization, labs, prescriptions, history and responsiveness were loaded as separate tables."),
        ("2. Standardize column names","Whitespace/formatting issues were standardized so the same variables could be referenced consistently."),
        ("3. Audit structure and IDs","Row counts, unique patients, duplicate rows and duplicate patient IDs were checked. Prescription repeats were treated as expected one-to-many structure."),
        ("4. Align the patient cohort","Patient IDs were compared across the patient-level tables. An unwanted demography record with inpatient_number = 5 was identified and removed in the cleaning workflow."),
        ("5. Validate demographics","Impossible weight/height values were set to missing and BMI was recalculated from weight and height where possible."),
        ("6. Validate vital signs","Placeholder zeros in pulse, respiration and blood pressure were treated as missing; inverted systolic/diastolic BP was checked."),
        ("7. Validate cardiac measurements","LVEDD, mitral E/A-related measurements and tricuspid pressure were inspected for implausible values/unit issues; E/A could be derived where source measurements supported it."),
        ("8. Standardize categorical values","Known casing and text-coded flags were standardized, including occupation and type-II respiratory-failure coding."),
        ("9. Correct data types","Dates, ordered age categories and nullable binary/score fields were converted to appropriate types."),
        ("10. Audit missingness","True clinical missingness was retained. The cleaning notebook explicitly avoids automatically converting missing clinical measurements into normal values."),
        ("11. Preserve plausible extremes","Clinically severe biomarker values were not automatically removed just because they were statistical outliers."),
        ("12. Reshape prescriptions","Prescription records were handled so repeated drug rows did not duplicate patients in the final master table."),
        ("13. Validate outcome consistency","Death, discharge and readmission timing relationships were checked for internal consistency."),
        ("14. Final master-table audit","The target was one row per patient with a unique inpatient_number before analysis."),
    ]
    for title,desc in steps:
        with st.expander(title):
            st.write(desc)

    st.subheader("Current cleaned-data audit")
    audit = pd.DataFrame([{
        "Rows": len(df),
        "Columns": df.shape[1],
        "Unique patients": df["inpatient_number"].nunique() if "inpatient_number" in df.columns else np.nan,
        "Duplicate rows": int(df.duplicated().sum()),
        "Duplicate patient IDs": int(df["inpatient_number"].duplicated().sum()) if "inpatient_number" in df.columns else np.nan,
        "Missing cells (%)": round(df.isna().mean().mean()*100,1)
    }])
    st.dataframe(audit,use_container_width=True,hide_index=True)

    st.subheader("Highest-missing variables")
    miss = df.isna().mean().mul(100).sort_values(ascending=False).head(20).reset_index()
    miss.columns=["Column","Missing (%)"]
    st.plotly_chart(px.bar(miss,y="Column",x="Missing (%)",orientation="h"),use_container_width=True)
    st.info("Important: high missingness does not automatically mean the variable is wrong. Some specialized tests were only obtained for selected patients. For predictive modelling, imputation should occur inside the training pipeline after the train/test split.")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
elif page == "🧬 Feature Engineering":
    st.title("Feature Engineering")
    st.write("Derived variables convert raw clinical measurements into interpretable analysis features. These are calculated in the dashboard without changing the stored Excel file.")

    engineered = [
        ("BMI Category","Underweight / Normal / Overweight / Obese from BMI."),
        ("Obesity Flag","BMI ≥ 30."),
        ("Inflammation Flag","Any available hs-CRP > 5, WBC > 10 or NLR > 6."),
        ("Low Albumin Flag","Albumin < 35."),
        ("Inflammation + Albumin Group","Neither / Inflamed only / Low albumin only / Both."),
        ("Cardiac-Renal Group","Creatinine above median and BNP above median combinations."),
        ("CCI Band","0–2, 3–4, 5+ bands."),
        ("Shock Index","Pulse ÷ systolic blood pressure."),
        ("Medication Burden","Count of detected prescription-flag columns."),
        ("In-hospital Death","Derived from outcome_during_hospitalization = Dead when available."),
    ]
    for name,desc in engineered:
        st.markdown(f"**{name}:** {desc}")

    c1,c2 = st.columns(2)
    with c1:
        if "BMI Category" in df:
            b = df["BMI Category"].value_counts(dropna=False).reset_index(); b.columns=["BMI Category","Patients"]
            st.plotly_chart(px.bar(b,x="BMI Category",y="Patients"),use_container_width=True)
    with c2:
        if "Inflammation + Albumin Group" in df:
            g = df["Inflammation + Albumin Group"].value_counts(dropna=False).reset_index(); g.columns=["Group","Patients"]
            st.plotly_chart(px.bar(g,x="Group",y="Patients"),use_container_width=True)

    if "Cardiac-Renal Group" in df:
        st.subheader("Cardiac–renal feature")
        temp = df.groupby("Cardiac-Renal Group",dropna=False).agg(Patients=("Cardiac-Renal Group","size")).reset_index()
        st.dataframe(temp,use_container_width=True,hide_index=True)

# ============================================================
# DESCRIPTIVE ANALYSIS
# ============================================================
elif page == "📈 Descriptive Analysis":
    st.title("Descriptive Analysis")
    st.write("Descriptive analysis answers: **Who are the patients? What clinical patterns are present? How do outcomes vary across patient groups?**")

    with st.expander("Global filters",expanded=True):
        f = df.copy()
        c1,c2,c3,c4 = st.columns(4)
        if gender_col:
            vals = sorted(df[gender_col].dropna().astype(str).unique())
            sg = c1.multiselect("Gender",vals,default=vals)
            f = f[f[gender_col].astype(str).isin(sg)] if sg else f.iloc[0:0]
        if agecat_col:
            vals = sorted(df[agecat_col].dropna().astype(str).unique())
            sa = c2.multiselect("Age category",vals,default=vals)
            f = f[f[agecat_col].astype(str).isin(sa)] if sa else f.iloc[0:0]
        if nyha_col:
            vals = sorted(df[nyha_col].dropna().astype(str).unique())
            sn = c3.multiselect("NYHA",vals,default=vals)
            f = f[f[nyha_col].astype(str).isin(sn)] if sn else f.iloc[0:0]
        if killip_col:
            vals = sorted(df[killip_col].dropna().astype(str).unique())
            sk = c4.multiselect("Killip",vals,default=vals)
            f = f[f[killip_col].astype(str).isin(sk)] if sk else f.iloc[0:0]
        st.caption(f"Filtered patients: {len(f):,}")

    tab1,tab2,tab3,tab4 = st.tabs(["Population","Severity","Biomarkers","Outcomes"])
    with tab1:
        c1,c2 = st.columns(2)
        with c1:
            if gender_col:
                g=f[gender_col].value_counts().reset_index(); g.columns=["Gender","Patients"]
                st.plotly_chart(px.pie(g,names="Gender",values="Patients",hole=.45),use_container_width=True)
        with c2:
            if bmi_col:
                st.plotly_chart(px.histogram(f,x=bmi_col,nbins=30,marginal="box"),use_container_width=True)
    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            if nyha_col:
                t=f[nyha_col].value_counts().reset_index(); t.columns=["NYHA","Patients"]
                st.plotly_chart(px.bar(t,x="NYHA",y="Patients"),use_container_width=True)
        with c2:
            if killip_col:
                t=f[killip_col].value_counts().reset_index(); t.columns=["Killip","Patients"]
                st.plotly_chart(px.bar(t,x="Killip",y="Patients"),use_container_width=True)
        if nyha_col and killip_col and in_hospital_col:
            z=f.pivot_table(index=nyha_col,columns=killip_col,values=in_hospital_col,aggfunc="mean")*100
            st.plotly_chart(px.imshow(z,text_auto=".1f",aspect="auto",labels={"color":"In-hospital death (%)"}),use_container_width=True)
    with tab3:
        selected = st.selectbox("Select any available biomarker / clinical measurement",list(biomarkers.keys())) if biomarkers else None
        if selected:
            c=biomarkers[selected]
            s=pd.to_numeric(f[c],errors="coerce")
            st.plotly_chart(px.histogram(pd.DataFrame({selected:s}).dropna(),x=selected,nbins=35,marginal="box"),use_container_width=True)
            st.caption(f"Column used: {c}")
    with tab4:
        rows=[]
        for label,col in [("In-hospital death",in_hospital_col),("28-day death",death28_col),("3-month death",death3m_col),("6-month death",death6m_col),("28-day readmission",readm28_col),("3-month readmission",readm3m_col),("6-month readmission",readm6m_col),("6-month ED return",ed6m_col)]:
            if col: rows.append((label,outcome_rate(f,col)))
        if rows:
            t=pd.DataFrame(rows,columns=["Outcome","Rate"])
            st.plotly_chart(px.bar(t,x="Outcome",y="Rate",text=t.Rate.map(lambda x:f"{x:.1f}%")),use_container_width=True)

# ============================================================
# PRESCRIPTIVE ANALYSIS — ALL 30 QUESTIONS
# ============================================================
elif page == "💡 Prescriptive Analysis":
    st.title("Prescriptive / Clinical Question Explorer")
    st.write("The project defined 30 clinical questions. This page keeps the exact question framing and provides an interactive evidence view where the required variables are available in the cleaned dataset.")

    questions = [
        (1,"Cardiac severity","Does a higher NYHA class lead to more readmissions and deaths at 28 days, 3 months and 6 months, and which classes need closer follow-up after discharge?"),
        (2,"Cardiac severity","At which Killip grade does death in hospital and within 28 days rise enough to need priority clinical review?"),
        (3,"Cardiac severity","Are patients with an enlarged left ventricle, especially those with an abnormal filling pattern, more likely to be readmitted or die within 6 months?"),
        (4,"Cardiac severity","Do left-sided, right-sided and both-sided heart failure differ in lung pressure, BNP and outcomes, and should they be monitored differently?"),
        (5,"Cardiac severity","Do patients with a prior heart attack or peripheral vascular disease show more heart-muscle damage and worse outcomes, and are they receiving aspirin and a statin?"),
        (6,"Blood markers","At what BNP level does readmission and death risk rise clearly enough to use it as a review trigger?"),
        (7,"Blood markers","Does heart-muscle injury at admission (high troponin, CK-MB, myoglobin) identify patients at higher risk of dying in hospital or within 28 days?"),
        (8,"Blood markers","Do high inflammation markers (hs-CRP, WBC, NLR) together with low albumin identify patients at higher risk of death, and should inflammation be part of routine risk review?"),
        (9,"Blood markers","Are anemic patients (low hemoglobin, high RDW) in worse NYHA classes and at higher risk of readmission and death?"),
        (10,"Electrolytes","Which sodium and potassium ranges are linked with higher death and readmission, and are patients on more diuretics more likely to fall into these ranges?"),
        (11,"Coagulation","Is a high D-dimer linked to higher death, and are patients on blood thinners kept within a safe INR range?"),
        (12,"Liver / congestion","Do liver markers of congestion (bilirubin, AST/ALT, low albumin) rise with lung pressure, and do they predict worse outcomes?"),
        (13,"Blood gas","Among patients who had a blood-gas test, do high lactate, low pH and low oxygen saturation identify those most likely to die in hospital?"),
        (14,"Kidney","Does worse kidney function (lower eGFR stage, higher creatinine, urea and cystatin) increase readmission and death?"),
        (15,"Heart + kidney","Which patients have both kidney problems and abnormal heart markers, and how much higher is their risk than patients with only one problem?"),
        (16,"Kidney","Is acute kidney failure during admission more dangerous than long-standing kidney disease alone?"),
        (17,"Comorbidity","Does readmission and death risk rise step by step as the CCI score increases?"),
        (18,"Comorbidity + medication","Which patients combine high comorbidity, high medication count and poor outcomes, and how many should be referred for multidisciplinary review?"),
        (19,"Respiratory","Do patients with COPD or type II respiratory failure have higher CO2 levels, more oxygen needs and more readmissions?"),
        (20,"Respiratory","How do patients who needed ventilation differ in heart severity, blood markers and outcomes from those who did not?"),
        (21,"Neurology","Are patients with dementia, stroke history, paralysis or reduced consciousness more likely to be discharged to non-home settings and to die?"),
        (22,"Medication","Do patients who receive more guideline heart failure medicines have lower readmission and death?"),
        (23,"Medication","Do patients who needed heart-strengthening drugs form an advanced heart failure group with worse outcomes?"),
        (24,"Vitals","Does low blood pressure or a high shock index (pulse ÷ systolic BP) at admission predict death?"),
        (25,"Nutrition / body composition","Are underweight patients at higher risk than overweight patients, and does low albumin explain this?"),
        (26,"Demographics","Which age and gender groups have the highest 6-month death after taking heart severity and comorbidity into account?"),
        (27,"Hospital process","Do emergency admissions stay longer and have worse outcomes than planned admissions?"),
        (28,"Hospital process","Are patients with very short hospital stays readmitted sooner, suggesting they may have been discharged too early?"),
        (29,"Hospital process","Which discharge destinations have the highest readmission and emergency-return rates?"),
        (30,"Frequent returners","Who are the frequent returners (multiple visits or ED return within 6 months), and what profile do they share?"),
    ]
    qmap={f"Q{n}. {text}":(n,domain,text) for n,domain,text in questions}
    selected_q=st.selectbox("Select a clinical question",list(qmap.keys()))
    qn,domain,qtext=qmap[selected_q]
    st.markdown(f"### Q{qn} — {domain}")
    st.info(qtext)

    # Generic variable explorer works for all 30 questions and is intentionally transparent.
    available = list(biomarkers.keys())
    default_var = available[0] if available else None
    if default_var:
        c1,c2=st.columns(2)
        with c1: var=st.selectbox("Primary variable",available,index=0)
        with c2:
            outcome_options=[]
            for label,col in [("In-hospital death",in_hospital_col),("28-day death",death28_col),("3-month death",death3m_col),("6-month death",death6m_col),("28-day readmission",readm28_col),("3-month readmission",readm3m_col),("6-month readmission",readm6m_col),("6-month ED return",ed6m_col)]:
                if col: outcome_options.append((label,col))
            outcome_label=st.selectbox("Outcome",[x[0] for x in outcome_options]) if outcome_options else None
        if outcome_label:
            outcome_col=dict(outcome_options)[outcome_label]
            temp=df[[biomarkers[var],outcome_col]].copy()
            temp[biomarkers[var]]=pd.to_numeric(temp[biomarkers[var]],errors="coerce")
            temp[outcome_col]=pd.to_numeric(temp[outcome_col],errors="coerce")
            temp=temp.dropna()
            if len(temp)>=10:
                temp["Group"]=pd.qcut(temp[biomarkers[var]],q=4,duplicates="drop")
                summary=temp.groupby("Group",observed=True)[outcome_col].agg(["count","mean"]).reset_index()
                summary["Rate (%)"]=summary["mean"]*100
                st.plotly_chart(px.bar(summary,x="Group",y="Rate (%)",text=summary["Rate (%)"].map(lambda x:f"{x:.1f}%"),labels={"Group":var}),use_container_width=True)
                st.dataframe(summary.drop(columns="mean"),use_container_width=True,hide_index=True)
                st.caption("Quartile groups are an exploratory view; they do not by themselves establish causality or a clinical treatment threshold.")
            else:
                st.warning("Not enough paired observations for a stable exploratory comparison.")

    if qn==8 and "Inflammation + Albumin Group" in df:
        st.subheader("Project-specific Q8 feature")
        if in_hospital_col:
            s=df.groupby("Inflammation + Albumin Group")[in_hospital_col].mean().mul(100).reset_index(name="In-hospital death (%)")
            st.plotly_chart(px.bar(s,x="Inflammation + Albumin Group",y="In-hospital death (%)",text=s["In-hospital death (%)"].map(lambda x:f"{x:.1f}%")),use_container_width=True)

    if qn==15 and "Cardiac-Renal Group" in df:
        st.subheader("Project-specific Q15 feature")
        out=death6m_col or in_hospital_col
        if out:
            s=df.groupby("Cardiac-Renal Group")[out].mean().mul(100).reset_index(name="Death rate (%)")
            st.plotly_chart(px.bar(s,x="Cardiac-Renal Group",y="Death rate (%)",text=s["Death rate (%)"].map(lambda x:f"{x:.1f}%")),use_container_width=True)

    st.warning("Prescriptive statements in the source questions are clinical review ideas. The dashboard presents observed associations and should not be interpreted as proving that a treatment caused an outcome.")

# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================
elif page == "🤖 Predictive Analytics":
    st.title("Predictive Analytics")
    st.write("This section demonstrates how the cleaned patient-level data can be used for mortality prediction. The model is for research/analytical use and is not a clinical diagnosis.")

    target_options=[]
    for label,col in [("In-hospital death",in_hospital_col),("28-day death",death28_col),("6-month death",death6m_col),("6-month readmission",readm6m_col)]:
        if col: target_options.append((label,col))
    if not target_options:
        st.warning("No binary outcome column was detected for modelling.")
    else:
        target_label=st.selectbox("Prediction target",[x[0] for x in target_options])
        target=dict(target_options)[target_label]

        candidate=[]
        for label,col in [("BMI",bmi_col),("Gender",gender_col),("Age category",agecat_col),("NYHA",nyha_col),("Killip",killip_col),("HF type",hf_type_col)]+[(k,v) for k,v in biomarkers.items()]:
            if col and col not in candidate: candidate.append(col)
        # Remove outcome-derived variables and post-outcome variables.
        excluded={target,in_hospital_col,death28_col,death3m_col,death6m_col,readm28_col,readm3m_col,readm6m_col,ed6m_col,los_col,readm_time_col,ed_time_col,outcome_text_col}
        candidate=[c for c in candidate if c and c not in excluded]
        labels={c:(next((k for k,v in biomarkers.items() if v==c),None) or c) for c in candidate}
        selected=st.multiselect("Model features",candidate,default=candidate[:min(12,len(candidate))],format_func=lambda c:labels[c])

        if len(selected)>=2:
            m=df[selected+[target]].copy()
            y=pd.to_numeric(m[target],errors="coerce")
            valid=y.isin([0,1])
            m=m.loc[valid].copy(); y=y.loc[valid].astype(int)
            X=m[selected].copy()
            cat=[c for c in selected if not pd.api.types.is_numeric_dtype(X[c])]
            num=[c for c in selected if c not in cat]
            if y.nunique()<2:
                st.warning("The selected outcome does not contain both outcome classes.")
            else:
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
                pre=ColumnTransformer([
                    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
                ],remainder="drop")
                models={
                    "Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced"),
                    "Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"),
                    "ANN":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,random_state=42)
                }
                results=[]; curves={}
                for name,model in models.items():
                    pipe=Pipeline([("prep",pre),("model",model)])
                    pipe.fit(X_train,y_train)
                    pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
                    results.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),"ROC-AUC":roc_auc_score(y_test,prob),"PR-AUC":average_precision_score(y_test,prob)})
                    fpr,tpr,_=roc_curve(y_test,prob); curves[name]=(fpr,tpr,roc_auc_score(y_test,prob))
                res=pd.DataFrame(results).sort_values("ROC-AUC",ascending=False)
                st.dataframe(res.style.format({c:"{:.3f}" for c in res.columns if c!="Model"}),use_container_width=True,hide_index=True)
                st.subheader("ROC comparison")
                fig=go.Figure()
                for name,(fpr,tpr,auc) in curves.items(): fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{name} (AUC {auc:.3f})"))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random"))
                fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
                st.plotly_chart(fig,use_container_width=True)
                st.info("Because mortality/readmission outcomes can be imbalanced, ROC-AUC should be read together with PR-AUC, recall and the confusion matrix. Model performance here is recomputed from the current cleaned file and selected predictors.")
        else:
            st.info("Select at least two predictors to run the comparison.")

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":
    st.title("Model Performance")
    st.write("Use this page to compare the current model run with the project-level model results reported in the analysis work.")
    st.subheader("Previously reported model comparison")
    prior = pd.DataFrame({"Model":["Logistic Regression","Random Forest","ANN"],"ROC-AUC":[.792,.797,.519],"PR-AUC":[.251,.191,.021]})
    st.dataframe(prior,use_container_width=True,hide_index=True)
    st.caption("These values are the previously reported analysis results and should be interpreted in the context of their original target, feature set, split and sample. The predictive page above recomputes a model using the current dashboard data and selected features.")
    st.subheader("What the metrics mean")
    st.markdown("- **ROC-AUC:** discrimination across classification thresholds.\n- **PR-AUC:** especially informative when the positive outcome is uncommon.\n- **Recall:** proportion of actual positive cases detected.\n- **Precision:** proportion of predicted positives that are truly positive.\n- **F1:** balance between precision and recall.")

# ============================================================
# PATIENT EXPLORER
# ============================================================
elif page == "🔎 Patient Explorer":
    st.title("Clinical Explorer")
    st.write("Explore the cleaned dataset across many criteria instead of being limited to two biomarkers.")
    f=df.copy()
    c1,c2,c3,c4=st.columns(4)
    if gender_col:
        vals=sorted(df[gender_col].dropna().astype(str).unique()); s=c1.multiselect("Gender",vals,default=vals); f=f[f[gender_col].astype(str).isin(s)] if s else f.iloc[0:0]
    if agecat_col:
        vals=sorted(df[agecat_col].dropna().astype(str).unique()); s=c2.multiselect("Age category",vals,default=vals); f=f[f[agecat_col].astype(str).isin(s)] if s else f.iloc[0:0]
    if nyha_col:
        vals=sorted(df[nyha_col].dropna().astype(str).unique()); s=c3.multiselect("NYHA",vals,default=vals); f=f[f[nyha_col].astype(str).isin(s)] if s else f.iloc[0:0]
    if killip_col:
        vals=sorted(df[killip_col].dropna().astype(str).unique()); s=c4.multiselect("Killip",vals,default=vals); f=f[f[killip_col].astype(str).isin(s)] if s else f.iloc[0:0]
    st.write(f"**{len(f):,} patients** match the selected criteria.")
    selected=st.selectbox("Variable to inspect",list(biomarkers.keys())) if biomarkers else None
    if selected:
        c=biomarkers[selected]
        s=pd.to_numeric(f[c],errors="coerce")
        c1,c2=st.columns(2)
        with c1:
            metric_card("🧪",selected,f"n={s.notna().sum():,}")
            metric_card("📌","Median",f"{s.median():.2f}" if s.notna().any() else "N/A")
        with c2:
            st.plotly_chart(px.histogram(pd.DataFrame({selected:s}).dropna(),x=selected,nbins=35,marginal="box"),use_container_width=True)
    show_cols=[c for c in ["inpatient_number",gender_col,agecat_col,bmi_col,nyha_col,killip_col,hf_type_col]+list(biomarkers.values()) if c and c in f.columns]
    st.dataframe(f[show_cols].head(200),use_container_width=True,hide_index=True)

# ============================================================
# INSIGHTS & KEY TAKEAWAYS
# ============================================================
elif page == "⭐ Insights & Key Takeaways":
    st.title("Insights & Key Takeaways")
    st.write("This page separates **data-derived findings** from broader project interpretation. Values are calculated from the cleaned file where possible.")

    # Dynamic findings
    findings=[]
    if in_hospital_col:
        findings.append(f"The current cleaned cohort contains **{len(df):,} patient rows**, with an in-hospital death rate of **{outcome_rate(df,in_hospital_col):.1f}%** based on the available in-hospital outcome definition.")
    if death28_col:
        findings.append(f"The observed 28-day death rate in the current file is **{outcome_rate(df,death28_col):.1f}%**.")
    if readm6_col:
        findings.append(f"The observed 6-month readmission rate in the current file is **{outcome_rate(df,readm6_col):.1f}%**.")
    if "Inflammation + Albumin Group" in df.columns and death6m_col:
        g=df.groupby("Inflammation + Albumin Group")[death6m_col].mean().dropna()*100
        if len(g)>=2:
            hi=g.idxmax(); findings.append(f"For the engineered inflammation/albumin grouping, the **{hi}** group has the highest observed 6-month death rate in the current file ({g.max():.1f}%). This is an association, not proof of causation.")
    if "Cardiac-Renal Group" in df.columns and death6m_col:
        g=df.groupby("Cardiac-Renal Group")[death6m_col].mean().dropna()*100
        if len(g)>=2:
            hi=g.idxmax(); findings.append(f"For the cardiac–renal grouping, **{hi}** has the highest observed 6-month death rate in the current file ({g.max():.1f}%).")
    if biomarkers:
        findings.append(f"The cleaned dataset exposes **{len(biomarkers)}** mapped clinical/laboratory variables for interactive exploration, rather than limiting the dashboard to two biomarkers.")

    for x in findings:
        st.markdown(f'<div class="takeaway">{x}</div>',unsafe_allow_html=True)

    st.subheader("Key takeaways from the project")
    takeaways=[
        "The project integrates seven source domains into a patient-level heart-failure dataset so clinical, laboratory, cardiac, medication and outcome information can be examined together.",
        "Data cleaning is clinically conservative: true missing measurements are not automatically converted to normal values, and plausible severe laboratory values are not removed simply because they are statistical extremes.",
        "Feature engineering creates clinically interpretable groups such as BMI category, inflammation + albumin, cardiac–renal burden, CCI bands, shock index and medication burden.",
        "The prescriptive analysis contains 30 clinical questions spanning cardiac severity, biomarkers, kidney function, comorbidity, respiratory/neurological status, medication, vitals and hospital processes.",
        "The predictive layer compares Logistic Regression, Random Forest and ANN models and reports both ROC-AUC and PR-AUC because outcome imbalance matters.",
        "Observed associations should be used as risk-review signals and hypothesis-generating evidence, not as proof that a treatment or clinical action caused a better outcome."
    ]
    for i,t in enumerate(takeaways,1):
        st.markdown(f"**{i}.** {t}")

# ============================================================
# CONCLUSION
# ============================================================
elif page == "🏁 Conclusion":
    st.title("Project Conclusion")
    st.markdown("""
    <div class="section-card">
    <h3>Overall conclusion</h3>
    <p>This project creates an end-to-end clinical analytics workflow for heart-failure patients: the seven source datasets are cleaned and integrated into a patient-level master dataset, clinically meaningful features are engineered, descriptive patterns are explored, 30 prescriptive questions are examined, and predictive models are evaluated for mortality/readmission outcomes.</p>
    <p>The dashboard brings these stages together so a user can move from <b>data quality → patient profile → clinical patterns → outcome associations → predictive performance → key takeaways</b>.</p>
    </div>
    """,unsafe_allow_html=True)

    st.subheader("What the project contributes")
    for x in [
        "A reproducible patient-level data-cleaning workflow.",
        "A broad clinical feature set covering demographics, cardiac severity, biomarkers, kidney function, inflammation/nutrition, comorbidities, medication and hospital process.",
        "A structured set of 30 prescriptive clinical questions.",
        "An interactive predictive layer comparing Logistic Regression, Random Forest and ANN approaches.",
        "A single dashboard that communicates technical analysis in a format intended for hospital/research users."
    ]:
        st.markdown(f"✓ {x}")

    st.warning("This dashboard is for research and analytical use. Associations in observational data do not establish causality, and model predictions should not be treated as a clinical diagnosis or treatment recommendation.")

# ============================================================
# END
# ============================================================
