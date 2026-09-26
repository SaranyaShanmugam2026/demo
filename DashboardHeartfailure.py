
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HeartCare Clinical Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# HOSPITAL-STYLE FRONT END
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: #F5F8FA;
    }

    [data-testid="stSidebar"] {
        background: #073B4C;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hospital-header {
        background: linear-gradient(135deg, #087F5B, #073B4C);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 6px 20px rgba(0,0,0,.08);
    }

    .hospital-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 750;
    }

    .hospital-header p {
        margin: 7px 0 0 0;
        font-size: 15px;
        opacity: .92;
    }

    .section-card {
        background: white;
        padding: 20px 22px;
        border-radius: 15px;
        box-shadow: 0 3px 14px rgba(0,0,0,.05);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 15px;
        border-left: 5px solid #087F5B;
        min-height: 115px;
        box-shadow: 0 3px 14px rgba(0,0,0,.05);
    }

    .metric-icon {
        font-size: 25px;
    }

    .metric-title {
        color: #637B83;
        font-size: 13px;
        font-weight: 650;
    }

    .metric-value {
        color: #073B4C;
        font-size: 27px;
        font-weight: 750;
        margin-top: 4px;
    }

    .takeaway {
        background: #EAF5F8;
        border-left: 5px solid #087F9B;
        padding: 15px 18px;
        border-radius: 9px;
        margin: 12px 0;
    }

    .review {
        background: #FFF8E8;
        border-left: 5px solid #D89B00;
        padding: 15px 18px;
        border-radius: 9px;
        margin: 12px 0;
    }

    .research-note {
        background: #F2F4F6;
        border-left: 5px solid #637B83;
        padding: 13px 16px;
        border-radius: 9px;
        margin: 12px 0;
        color: #43535A;
        font-size: 13px;
    }

    .question-box {
        background: white;
        border: 1px solid #DCE5E9;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 15px;
    }

    .small-muted {
        color: #637B83;
        font-size: 13px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
DATA_CANDIDATES = [
    "Cardiac_Cleaned_Data.xlsb",
    "Cardiac_Cleaned_Data.xlsx",
    "Cardiac_Cleaned_Data.csv"
]

@st.cache_data
def load_repo_data():
    errors = []
    for file in DATA_CANDIDATES:
        try:
            if file.endswith(".xlsb"):
                return pd.read_excel(file, engine="pyxlsb")
            if file.endswith(".xlsx"):
                return pd.read_excel(file)
            return pd.read_csv(file)
        except Exception as e:
            errors.append(f"{file}: {e}")
    raise FileNotFoundError(" | ".join(errors))

uploaded = None
with st.sidebar:
    st.markdown("### Data source")
    uploaded = st.file_uploader(
        "Optional: upload cleaned dataset",
        type=["csv", "xlsx", "xlsb"]
    )

if uploaded is not None:
    if uploaded.name.endswith(".xlsb"):
        df = pd.read_excel(uploaded, engine="pyxlsb")
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
else:
    try:
        df = load_repo_data()
    except Exception as e:
        st.error("The cleaned dataset could not be loaded.")
        st.code(str(e))
        st.info(
            "Place Cardiac_Cleaned_Data.xlsb beside this Python file, "
            "or upload the cleaned dataset using the sidebar."
        )
        st.stop()

# ============================================================
# BASIC CLEANUP / DERIVED VARIABLES USED BY THE DASHBOARD
# ============================================================
df = df.copy()

def numeric(col):
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

# NLR is created only if it is not already present.
if "nlr" not in df.columns and {"neutrophil_count", "lymphocyte_count"}.issubset(df.columns):
    den = numeric("lymphocyte_count")
    df["nlr"] = numeric("neutrophil_count") / den.replace(0, np.nan)

if "in_hospital_death" not in df.columns and "outcome_during_hospitalization" in df.columns:
    df["in_hospital_death"] = (
        df["outcome_during_hospitalization"].astype(str).str.strip().str.lower()
        == "dead"
    ).astype(int)

# ============================================================
# COLUMN HELPERS
# ============================================================
def first_existing(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def existing(candidates):
    return [c for c in candidates if c in df.columns]

age_col = first_existing(["age", "age_years"])
agecat_col = first_existing(["agecat"])
gender_col = first_existing(["gender", "sex"])
bmi_col = first_existing(["bmi"])
nyha_col = first_existing(["nyha_cardiac_function_classification"])
killip_col = first_existing(["killip_grade"])
hf_type_col = first_existing(["type_of_heart_failure"])
cci_col = first_existing(["cci_score"])
total_drugs_col = first_existing(["total_drugs"])
mortality_in_col = first_existing(["in_hospital_death"])
mortality28_col = first_existing(["death_within_28_days"])
mortality3_col = first_existing(["death_within_3_months"])
mortality6_col = first_existing(["death_within_6_months"])
readm28_col = first_existing(["re_admission_within_28_days"])
readm3_col = first_existing(["re_admission_within_3_months"])
readm6_col = first_existing(["re_admission_within_6_months"])
ed6_col = first_existing(["return_to_emergency_department_within_6_months"])

OUTCOMES = {}
if mortality_in_col: OUTCOMES["In-hospital mortality"] = mortality_in_col
if mortality28_col: OUTCOMES["28-day mortality"] = mortality28_col
if mortality3_col: OUTCOMES["3-month mortality"] = mortality3_col
if mortality6_col: OUTCOMES["6-month mortality"] = mortality6_col
if readm28_col: OUTCOMES["28-day readmission"] = readm28_col
if readm3_col: OUTCOMES["3-month readmission"] = readm3_col
if readm6_col: OUTCOMES["6-month readmission"] = readm6_col
if ed6_col: OUTCOMES["6-month ED return"] = ed6_col

BIOMARKERS = {
    "BNP": "brain_natriuretic_peptide",
    "High-sensitivity troponin": "high_sensitivity_troponin",
    "hs-CRP": "hs_crp",
    "WBC": "white_blood_cell",
    "NLR": "nlr",
    "Albumin": "albumin",
    "Hemoglobin": "hemoglobin",
    "RDW": "coefficient_of_variation_of_red_blood_cell_distribution_width",
    "Sodium": "sodium",
    "Potassium": "potassium",
    "D-dimer": "d_dimer",
    "Creatinine": "creatinine_enzymatic_method",
    "eGFR": "glomerular_filtration_rate",
    "Urea": "urea",
    "Uric acid": "uric_acid",
    "Cystatin": "cystatin",
    "Lactate": "lactate",
    "pH": "ph",
    "Oxygen saturation": "oxygen_saturation",
    "Total bilirubin": "total_bilirubin",
    "AST/ALT ratio": "ast_alt_ratio",
    "LVEF": "lvef",
    "LVEDD": "lvedd_mm",
    "Tricuspid pressure": "tricuspid_valve_return_pressure"
}
BIOMARKERS = {k:v for k,v in BIOMARKERS.items() if v in df.columns}

DOMAIN_VARIABLES = {
    "Demographics": existing([
        "agecat","gender","weight","height","bmi","occupation","bmi_category","obesity_flag"
    ]),
    "Cardiac severity": existing([
        "nyha_cardiac_function_classification","killip_grade",
        "type_of_heart_failure","lvef","lvedd_mm","ea"
    ]),
    "Cardiac biomarkers": existing([
        "brain_natriuretic_peptide","high_sensitivity_troponin","creatine_kinase_isoenzyme",
        "myoglobin","tricuspid_valve_return_pressure"
    ]),
    "Inflammation & nutrition": existing([
        "hs_crp","white_blood_cell","nlr","neutrophil_count","lymphocyte_count",
        "albumin","hemoglobin","coefficient_of_variation_of_red_blood_cell_distribution_width"
    ]),
    "Kidney": existing([
        "glomerular_filtration_rate","creatinine_enzymatic_method","urea","uric_acid",
        "cystatin","acute_renal_failure","moderate_to_severe_chronic_kidney_disease"
    ]),
    "Electrolytes & blood gas": existing([
        "sodium","potassium","lactate","ph","oxygen_saturation",
        "partial_oxygen_pressure","partial_pressure_of_carbon_dioxide","fio2"
    ]),
    "Respiratory / hemodynamics": existing([
        "systolic_blood_pressure","diastolic_blood_pressure","map_value","pulse",
        "respiration","respiratory_support","respiratory_support_flag",
        "type_ii_respiratory_failure","gcs","consciousness","body_temperature"
    ]),
    "Comorbidity": existing([
        "diabetes","moderate_to_severe_chronic_kidney_disease",
        "chronic_obstructive_pulmonary_disease","cerebrovascular_disease",
        "myocardial_infarction","dementia","peripheral_vascular_disease",
        "liver_disease","peptic_ulcer_disease","solid_tumor",
        "acute_renal_failure","cci_score","comorbidity_count"
    ]),
    "Medication": existing([
        "total_drugs","medication_burden","polypharmacy_flag",
        "guideline_directed_medication_count"
    ]),
    "Hospital course / discharge": existing([
        "admission_way","dischargeday","destinationdischarge",
        "visit_times","return_to_emergency_department_within_6_months",
        "readmission_time_days_from_admission",
        "time_to_emergency_department_within_6_months"
    ])
}

# ============================================================
# ANALYSIS LIBRARY: Q1-Q30
# ============================================================
PRESCRIPTIVE_Q = [
("Q1","BNP and post-discharge outcomes",
 "Does an elevated BNP level at admission identify patients who have a higher risk of readmission or mortality?",
 ["brain_natriuretic_peptide"],
 ["re_admission_within_28_days","re_admission_within_3_months","re_admission_within_6_months",
  "death_within_28_days","death_within_3_months","death_within_6_months"]),
("Q2","Cardiac injury biomarkers",
 "Do elevated troponin, CK-MB and myoglobin identify patients at higher short-term mortality risk?",
 ["high_sensitivity_troponin","creatine_kinase_isoenzyme","myoglobin"],
 ["in_hospital_death","death_within_28_days"]),
("Q3","Inflammation + albumin",
 "Do elevated inflammatory markers combined with low albumin identify higher-risk patients?",
 ["hs_crp","white_blood_cell","nlr","albumin"],
 ["death_within_28_days","death_within_6_months","re_admission_within_6_months"]),
("Q4","Anemia + RDW",
 "Are lower hemoglobin and higher RDW associated with more severe NYHA class and worse outcomes?",
 ["hemoglobin","coefficient_of_variation_of_red_blood_cell_distribution_width","nyha_cardiac_function_classification"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q5","Sodium + potassium",
 "Which sodium and potassium ranges are associated with mortality and readmission?",
 ["sodium","potassium"],
 ["re_admission_within_6_months","death_within_28_days","death_within_6_months"]),
("Q6","D-dimer",
 "Is elevated D-dimer associated with higher mortality?",
 ["d_dimer"],
 ["death_within_28_days","death_within_6_months"]),
("Q7","Liver / congestion markers",
 "Are bilirubin, AST/ALT and albumin associated with tricuspid pressure and worse outcomes?",
 ["total_bilirubin","ast_alt_ratio","albumin","tricuspid_valve_return_pressure"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q8","Blood gas markers",
 "Do elevated lactate, low pH and low oxygen saturation identify higher-risk patients?",
 ["lactate","ph","oxygen_saturation"],
 ["in_hospital_death","death_within_28_days","death_within_6_months"]),
("Q9","Kidney function",
 "Does worsening kidney function correspond to higher readmission and mortality?",
 ["glomerular_filtration_rate","creatinine_enzymatic_method","urea","cystatin"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q10","Cardiac + kidney burden",
 "Do patients with both abnormal kidney and cardiac markers experience worse outcomes?",
 ["creatinine_enzymatic_method","brain_natriuretic_peptide"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q11","NYHA severity",
 "Is increasing NYHA class associated with higher readmission and mortality?",
 ["nyha_cardiac_function_classification"],
 ["death_within_28_days","death_within_6_months","re_admission_within_6_months"]),
("Q12","Killip severity",
 "Is increasing Killip grade associated with higher in-hospital and 28-day mortality?",
 ["killip_grade"],
 ["in_hospital_death","death_within_28_days"]),
("Q13","LVEDD + E/A",
 "Are enlarged LVEDD and abnormal E/A filling associated with readmission or mortality?",
 ["lvedd_mm","ea"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q14","Heart-failure phenotype",
 "Do left-sided, right-sided and combined heart-failure phenotypes differ in clinical burden and outcomes?",
 ["type_of_heart_failure","tricuspid_valve_return_pressure","brain_natriuretic_peptide"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q15","CCI",
 "Does increasing Charlson Comorbidity Index correspond to progressively higher readmission and mortality?",
 ["cci_score"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q16","Acute kidney failure vs CKD",
 "Do acute kidney failure and chronic kidney disease groups differ in length of stay and outcomes?",
 ["acute_renal_failure","moderate_to_severe_chronic_kidney_disease","dischargeday"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q17","COPD / respiratory failure",
 "Are COPD and type II respiratory failure associated with worse gas exchange, oxygen needs and outcomes?",
 ["chronic_obstructive_pulmonary_disease","type_ii_respiratory_failure","partial_pressure_of_carbon_dioxide","oxygen_saturation"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q18","Mechanical ventilation / support",
 "How do patients requiring mechanical or non-invasive respiratory support differ from others?",
 ["respiratory_support_flag","oxygen_saturation","gcs","nyha_cardiac_function_classification","killip_grade"],
 ["in_hospital_death","death_within_28_days"]),
("Q19","Comorbidity + medication burden",
 "Are high comorbidity and high medication counts associated with worse outcomes?",
 ["cci_score","total_drugs"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q20","Guideline medication count",
 "Does the number of guideline-directed heart-failure medications at discharge relate to outcomes?",
 ["guideline_directed_medication_count"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q21","Inotrope use",
 "Do patients receiving inotropic medications have different severity or outcome profiles?",
 ["milrinone","dobutamine","deslanoside"],
 ["in_hospital_death","death_within_28_days"]),
("Q22","Blood pressure + shock index",
 "Are lower systolic blood pressure and higher admission shock index associated with mortality?",
 ["systolic_blood_pressure","pulse"],
 ["in_hospital_death","death_within_28_days"]),
("Q23","BMI + albumin",
 "Is lower BMI associated with lower albumin and worse outcomes?",
 ["bmi","albumin"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q24","Age + gender within severity strata",
 "After considering heart-failure severity and comorbidity burden, do age and gender groups differ in outcomes?",
 ["agecat","gender","nyha_cardiac_function_classification","cci_score"],
 ["death_within_6_months","re_admission_within_6_months"]),
("Q25","Emergency vs non-emergency admission",
 "Do emergency and non-emergency admissions differ in length of stay, readmission and mortality?",
 ["admission_way","dischargeday"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q26","Short hospital stay",
 "Is a very short hospital stay associated with earlier readmission or emergency return?",
 ["dischargeday","readmission_time_days_from_admission","time_to_emergency_department_within_6_months"],
 ["re_admission_within_6_months","return_to_emergency_department_within_6_months"]),
("Q27","Discharge destination",
 "Do readmission and emergency-return rates differ by discharge destination?",
 ["destinationdischarge"],
 ["re_admission_within_6_months","return_to_emergency_department_within_6_months"]),
("Q28","Neurologic / cognitive burden",
 "Are dementia, previous stroke, paralysis or reduced consciousness associated with non-home discharge and mortality?",
 ["dementia","cerebrovascular_disease","hemiplegia","consciousness"],
 ["death_within_6_months"]),
("Q29","Combined high-risk profile",
 "Which combination of biomarkers, severity, renal function, comorbidity, vitals and medication burden is associated with outcomes?",
 ["brain_natriuretic_peptide","high_sensitivity_troponin","hs_crp","albumin",
  "sodium","creatinine_enzymatic_method","glomerular_filtration_rate",
  "nyha_cardiac_function_classification","cci_score","lvef",
  "systolic_blood_pressure","pulse","total_drugs"],
 ["re_admission_within_6_months","death_within_6_months"]),
("Q30","Frequent returners",
 "What characteristics distinguish patients who repeatedly return to hospital or the emergency department?",
 ["visit_times","return_to_emergency_department_within_6_months","nyha_cardiac_function_classification",
  "cci_score","brain_natriuretic_peptide","albumin","sodium",
  "creatinine_enzymatic_method","glomerular_filtration_rate","total_drugs"],
 ["re_admission_within_6_months","return_to_emergency_department_within_6_months"])
]

PREDICTIVE_Q = [
("Q1","Demographics → in-hospital mortality",
 ["agecat","gender","weight","height","bmi","occupation"], "in_hospital_death"),
("Q2","Demographics + baseline clinical → 28-day mortality",
 ["agecat","gender","bmi","admission_way","nyha_cardiac_function_classification",
  "killip_grade","lvef","systolic_blood_pressure","pulse","cci_score"], "death_within_28_days"),
("Q3","NYHA vs Killip vs combined → 28-day mortality",
 ["nyha_cardiac_function_classification","killip_grade"], "death_within_28_days"),
("Q4","Renal markers → 6-month mortality / readmission",
 ["creatinine_enzymatic_method","urea","uric_acid","glomerular_filtration_rate",
  "cystatin","acute_renal_failure","moderate_to_severe_chronic_kidney_disease"], "death_within_6_months"),
("Q5","Cardiac biomarkers added to baseline → 6-month mortality",
 ["agecat","gender","pulse","systolic_blood_pressure",
  "nyha_cardiac_function_classification","killip_grade",
  "brain_natriuretic_peptide","high_sensitivity_troponin"], "death_within_6_months"),
("Q6","Inflammation + nutrition added to baseline → 6-month mortality",
 ["agecat","gender","pulse","systolic_blood_pressure",
  "nyha_cardiac_function_classification","killip_grade",
  "hs_crp","white_blood_cell","nlr","albumin"], "death_within_6_months"),
("Q7","Admission respiratory / neurologic profile → respiratory support",
 ["respiration","oxygen_saturation","partial_oxygen_pressure",
  "partial_pressure_of_carbon_dioxide","fio2","gcs","consciousness",
  "pulse","systolic_blood_pressure","body_temperature",
  "nyha_cardiac_function_classification","killip_grade"], "respiratory_support_flag"),
("Q8","Neurologic + comorbidity + severity → non-home discharge",
 ["dementia","cerebrovascular_disease","hemiplegia","gcs","consciousness",
  "cci_score","comorbidity_count","agecat","nyha_cardiac_function_classification",
  "killip_grade","respiratory_support_flag"], "destinationdischarge"),
("Q9","Strongest admission-time features → 28-day mortality",
 ["killip_grade","nyha_cardiac_function_classification","brain_natriuretic_peptide",
  "creatinine_enzymatic_method","urea","cystatin","cci_score","lactate","ph",
  "oxygen_saturation","lvedd_mm","ea","tricuspid_valve_return_pressure",
  "tricuspid_valve_return_velocity"], "death_within_28_days"),
("Q10","Combined clinical domains → 6-month mortality",
 ["killip_grade","nyha_cardiac_function_classification","brain_natriuretic_peptide",
  "high_sensitivity_troponin","hs_crp","albumin","sodium",
  "glomerular_filtration_rate","urea","cystatin","cci_score",
  "systolic_blood_pressure","pulse","age","bmi"], "death_within_6_months"),
("Q11","Risk groups from predicted 28-day mortality probabilities",
 ["killip_grade","nyha_cardiac_function_classification","brain_natriuretic_peptide",
  "creatinine_enzymatic_method","urea","cystatin","cci_score"], "death_within_28_days"),
("Q12","Logistic Regression vs Random Forest vs ANN",
 ["killip_grade","nyha_cardiac_function_classification","brain_natriuretic_peptide",
  "creatinine_enzymatic_method","urea","cystatin","cci_score"], "death_within_28_days"),
("Q13","Admission characteristics → 6-month readmission",
 ["nyha_cardiac_function_classification","killip_grade","systolic_blood_pressure",
  "pulse","respiration","glomerular_filtration_rate","urea","cystatin",
  "moderate_to_severe_chronic_kidney_disease","bnp_log","troponin_log",
  "nlr_log","albumin","hemoglobin","sodium","cci_score","diabetes",
  "chronic_obstructive_pulmonary_disease","age","male","bmi"], "re_admission_within_6_months"),
("Q14","Combined domains → 6-month mortality",
 ["nyha_cardiac_function_classification","killip_grade","bnp_log","troponin_log",
  "nlr_log","albumin","hemoglobin","sodium","glomerular_filtration_rate",
  "urea","cystatin","moderate_to_severe_chronic_kidney_disease","cci_score",
  "diabetes","chronic_obstructive_pulmonary_disease","myocardial_infarction",
  "type_ii_respiratory_failure","systolic_blood_pressure","pulse","respiration",
  "age","male","bmi","unconscious"], "death_within_6_months"),
("Q15","Previous cardiac history + current severity → in-hospital death",
 ["myocardial_infarction","congestive_heart_failure","peripheral_vascular_disease",
  "cerebrovascular_disease","right_or_both_hf",
  "nyha_cardiac_function_classification","killip_grade"], "in_hospital_death"),
("Q16","Inflammatory biomarkers → in-hospital / 28-day mortality",
 ["hs_crp","white_blood_cell","nlr"], "in_hospital_death")
]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def metric_card(icon, title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def clean_binary(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any():
        vals = sorted(x.dropna().unique())
        if set(vals).issubset({0,1}):
            return x
    mapping = {
        "yes":1, "y":1, "true":1, "dead":1, "death":1,
        "no":0, "n":0, "false":0, "alive":0, "survived":0
    }
    return s.astype(str).str.strip().str.lower().map(mapping)

def outcome_rate(data, col):
    if not col or col not in data.columns:
        return np.nan
    x = clean_binary(data[col])
    return x.mean() * 100

def pretty_col(c):
    return str(c).replace("_"," ").replace("  "," ").title()

def make_filtered_data():
    filtered = df.copy()

    # Age filter
    age_source = age_col if age_col else None
    if age_source:
        vals = numeric(age_source).dropna()
        if len(vals):
            lo, hi = int(np.floor(vals.min())), int(np.ceil(vals.max()))
            if lo < hi:
                selected = st.sidebar.slider("Age range", lo, hi, (lo, hi))
                filtered = filtered[
                    numeric(age_source).between(selected[0], selected[1])
                ]

    # Categorical filters
    for label, col in [
        ("Gender", gender_col),
        ("Age category", agecat_col),
        ("NYHA class", nyha_col),
        ("Killip grade", killip_col),
        ("Heart-failure type", hf_type_col),
        ("Admission way", first_existing(["admission_way"]))
    ]:
        if col and col in filtered.columns:
            vals = sorted(filtered[col].dropna().astype(str).unique().tolist())
            if vals and len(vals) <= 30:
                chosen = st.sidebar.multiselect(
                    label, vals, default=vals, key=f"filter_{col}"
                )
                if chosen:
                    filtered = filtered[filtered[col].astype(str).isin(chosen)]

    return filtered

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 4px 8px 4px;">
        <div style="font-size:48px;">❤️</div>
        <div style="font-size:24px;font-weight:800;">HeartCare AI</div>
        <div style="font-size:12px;opacity:.85;">Clinical Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏥 Executive Overview",
            "🔎 Clinical Explorer",
            "🧪 Biomarker Explorer",
            "❤️ Cardiac & Severity",
            "⚠️ Outcomes & Mortality",
            "💡 Prescriptive Analysis",
            "🤖 Predictive Analytics",
            "🧬 Feature Engineering",
            "🧹 Data Quality"
        ]
    )

    st.divider()
    st.caption(f"Dataset: {len(df):,} rows")
    st.caption("Research and analytical use only.")

# ============================================================
# GLOBAL FILTERS
# ============================================================
if page != "🧹 Data Quality":
    filtered_df = make_filtered_data()
else:
    filtered_df = df.copy()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hospital-header">
    <h1>❤️ Heart Failure Clinical Analytics</h1>
    <p>
        Patient population • biomarkers • cardiac severity • outcomes •
        prescriptive insights • predictive analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. EXECUTIVE OVERVIEW
# ============================================================
if page == "🏥 Executive Overview":

    st.subheader("Hospital-level snapshot")
    st.markdown(
        '<div class="small-muted">Use the sidebar filters to change the population shown throughout the dashboard.</div>',
        unsafe_allow_html=True
    )

    total = len(filtered_df)
    in_hosp = outcome_rate(filtered_df, mortality_in_col)
    death28 = outcome_rate(filtered_df, mortality28_col)
    death6 = outcome_rate(filtered_df, mortality6_col)
    readm6 = outcome_rate(filtered_df, readm6_col)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("👥","Patients",f"{total:,}")
    with c2: metric_card("🏥","In-hospital mortality",f"{in_hosp:.1f}%" if not np.isnan(in_hosp) else "N/A")
    with c3: metric_card("📅","28-day mortality",f"{death28:.1f}%" if not np.isnan(death28) else "N/A")
    with c4: metric_card("🗓️","6-month mortality",f"{death6:.1f}%" if not np.isnan(death6) else "N/A")
    with c5: metric_card("🔁","6-month readmission",f"{readm6:.1f}%" if not np.isnan(readm6) else "N/A")

    st.markdown("### What the dashboard covers")

    overview_items = [
        ("👥 Patient profile","Age, gender, BMI, occupation and body-composition patterns."),
        ("❤️ Cardiac severity","NYHA, Killip, heart-failure phenotype, LVEF and LVEDD."),
        ("🧪 Biomarkers","Cardiac injury, inflammation, nutrition, renal, electrolyte and blood-gas markers."),
        ("⚠️ Outcomes","In-hospital, 28-day, 3-month and 6-month mortality/readmission."),
        ("💡 Prescriptive analysis","The 30 analysis questions from the team's prescriptive notebooks."),
        ("🤖 Predictive analysis","The 16 predictive questions, including logistic regression, random forest and ANN."),
    ]
    cols = st.columns(3)
    for i,(title,desc) in enumerate(overview_items):
        with cols[i%3]:
            st.markdown(
                f'<div class="section-card"><b>{title}</b><br><span class="small-muted">{desc}</span></div>',
                unsafe_allow_html=True
            )

    # Outcome chart
    outcome_names = []
    outcome_rates = []
    for name,col in OUTCOMES.items():
        r = outcome_rate(filtered_df,col)
        if not np.isnan(r):
            outcome_names.append(name)
            outcome_rates.append(r)

    if outcome_names:
        plot_df = pd.DataFrame({"Outcome":outcome_names,"Rate":outcome_rates})
        fig = px.bar(
            plot_df, x="Outcome", y="Rate", text="Rate",
            title="Observed outcome rates in the selected population"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_yaxes(title="Patients (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="research-note"><b>Important:</b> These are observed dataset rates. They describe associations in this dataset and should not be interpreted as causal effects or clinical treatment recommendations.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 2. CLINICAL EXPLORER
# ============================================================
elif page == "🔎 Clinical Explorer":

    st.subheader("🔎 Clinical Explorer")
    st.write(
        "This is the main interactive area. Select a clinical domain, a variable, "
        "an outcome and the analyses you want to see."
    )

    domain = st.selectbox("Clinical domain", list(DOMAIN_VARIABLES.keys()))
    domain_vars = DOMAIN_VARIABLES[domain]

    if not domain_vars:
        st.warning("No variables from this domain are available in the loaded dataset.")
        st.stop()

    variable = st.selectbox(
        "Variable",
        domain_vars,
        format_func=pretty_col
    )

    outcome_name = st.selectbox(
        "Outcome",
        list(OUTCOMES.keys()) if OUTCOMES else ["No outcome columns detected"]
    )
    outcome_col = OUTCOMES.get(outcome_name)

    c1,c2,c3,c4 = st.columns(4)
    with c1: show_dist = st.checkbox("Show distribution", True)
    with c2: show_outcome = st.checkbox("Show outcome comparison", True)
    with c3: show_groups = st.checkbox("Show grouped analysis", True)
    with c4: show_takeaway = st.checkbox("Show key takeaway", True)

    data = filtered_df[[variable] + ([outcome_col] if outcome_col else [])].copy()

    if show_dist:
        st.markdown("### 1. Variable distribution")
        x = pd.to_numeric(data[variable], errors="coerce")
        if x.notna().sum() >= 5:
            fig = px.histogram(
                pd.DataFrame({pretty_col(variable):x.dropna()}),
                x=pretty_col(variable),
                nbins=30,
                marginal="box"
            )
        else:
            vc = data[variable].astype(str).value_counts().reset_index()
            vc.columns=["Category","Patients"]
            fig = px.bar(vc, x="Category", y="Patients")
        st.plotly_chart(fig, use_container_width=True)

    if show_outcome and outcome_col:
        st.markdown("### 2. Variable vs selected outcome")
        temp = data.copy()
        temp["_outcome"] = clean_binary(temp[outcome_col])

        numeric_x = pd.to_numeric(temp[variable], errors="coerce")
        if numeric_x.notna().sum() >= 5:
            temp["_x"] = numeric_x
            temp = temp.dropna(subset=["_x","_outcome"])
            if len(temp):
                fig = px.box(
                    temp, x="_outcome", y="_x",
                    points=False,
                    labels={"_outcome":"Outcome (0 = no event, 1 = event)",
                            "_x":pretty_col(variable)}
                )
                st.plotly_chart(fig, use_container_width=True)

                grouped = (
                    temp.groupby("_outcome")["_x"]
                    .agg(["count","median","mean"])
                    .reset_index()
                )
                grouped["Outcome"] = grouped["_outcome"].map({0:"No event",1:"Event"}).fillna(grouped["_outcome"].astype(str))
                grouped = grouped[["Outcome","count","median","mean"]]
                grouped.columns=["Outcome","Patients","Median","Mean"]
                st.dataframe(grouped, use_container_width=True, hide_index=True)
        else:
            ct = pd.crosstab(
                temp[variable].astype(str),
                temp["_outcome"],
                normalize="index"
            ) * 100
            ct = ct.rename(columns={0:"No event (%)",1:"Event (%)"})
            st.dataframe(ct.round(1), use_container_width=True)

    if show_groups:
        st.markdown("### 3. Simple high/low or quantile view")
        x = pd.to_numeric(data[variable], errors="coerce")
        if x.notna().sum() >= 10 and outcome_col:
            work = pd.DataFrame({"x":x, "outcome":clean_binary(data[outcome_col])}).dropna()
            try:
                work["Group"] = pd.qcut(work["x"], 4, duplicates="drop")
                g = work.groupby("Group", observed=True)["outcome"].agg(["count","mean"]).reset_index()
                g["Event rate (%)"] = g["mean"]*100
                fig = px.bar(
                    g, x="Group", y="Event rate (%)", text="Event rate (%)",
                    title=f"{outcome_name} across {pretty_col(variable)} quartiles"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    if show_takeaway:
        st.markdown("### 💡 Key takeaway")
        if outcome_col:
            rate = outcome_rate(filtered_df, outcome_col)
            median = pd.to_numeric(filtered_df[variable], errors="coerce").median()
            st.markdown(
                f'<div class="takeaway"><b>Selected population:</b> {len(filtered_df):,} patients. '
                f'<b>{outcome_name} rate:</b> {rate:.1f}% when available. '
                f'<b>Median {pretty_col(variable)}:</b> {median:.2f} when numeric. '
                f'This is a descriptive summary; it does not establish causation.</div>',
                unsafe_allow_html=True
            )

# ============================================================
# 3. BIOMARKER EXPLORER
# ============================================================
elif page == "🧪 Biomarker Explorer":

    st.subheader("🧪 Biomarker Explorer")
    st.write(
        "Every available biomarker can be selected from the dropdown. "
        "Choose the outcome and optional analyses you want to inspect."
    )

    if not BIOMARKERS:
        st.warning("No biomarker columns were detected.")
        st.stop()

    selected_name = st.selectbox("Biomarker", list(BIOMARKERS.keys()))
    biomarker = BIOMARKERS[selected_name]

    outcome_name = st.selectbox(
        "Outcome",
        list(OUTCOMES.keys())
    )
    outcome_col = OUTCOMES[outcome_name]

    a,b,c,d = st.columns(4)
    with a: show_distribution = st.checkbox("Distribution", True)
    with b: show_event = st.checkbox("Outcome comparison", True)
    with c: show_quartiles = st.checkbox("Quartile analysis", True)
    with d: show_missing = st.checkbox("Missingness", True)

    vals = numeric(biomarker)
    temp = pd.DataFrame({
        "Biomarker": vals,
        "Outcome": clean_binary(filtered_df[outcome_col])
    })

    if show_distribution:
        st.markdown("### Distribution")
        plot = temp["Biomarker"].dropna()
        if len(plot):
            fig = px.histogram(
                plot.to_frame(),
                x="Biomarker",
                nbins=35,
                marginal="box",
                title=f"{selected_name} distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_event:
        st.markdown("### Biomarker vs outcome")
        t = temp.dropna()
        if len(t):
            t["Outcome label"] = t["Outcome"].map({0:"No event",1:"Event"}).fillna("Unknown")
            fig = px.box(
                t, x="Outcome label", y="Biomarker",
                points=False,
                title=f"{selected_name} by {outcome_name}"
            )
            st.plotly_chart(fig, use_container_width=True)

    if show_quartiles:
        st.markdown("### Quartile analysis")
        t = temp.dropna()
        if len(t) >= 20:
            try:
                t["Quartile"] = pd.qcut(t["Biomarker"], 4, duplicates="drop")
                q = t.groupby("Quartile", observed=True)["Outcome"].agg(["count","mean"]).reset_index()
                q["Event rate (%)"] = q["mean"]*100
                fig = px.bar(
                    q, x="Quartile", y="Event rate (%)",
                    text="Event rate (%)",
                    title=f"{outcome_name} rate across {selected_name} quartiles"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    q.rename(columns={"count":"Patients"})[["Quartile","Patients","Event rate (%)"]].round(2),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                st.info("Quartile grouping could not be created for this biomarker.")

    if show_missing:
        miss = filtered_df[biomarker].isna().mean()*100
        st.markdown(
            f'<div class="review"><b>Data completeness:</b> {miss:.1f}% of the selected population has a missing {selected_name} value.</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="research-note"><b>How to use this:</b> This explorer is intentionally descriptive. '
        'It helps you understand each biomarker before feature engineering and predictive modeling.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 4. CARDIAC & SEVERITY
# ============================================================
elif page == "❤️ Cardiac & Severity":

    st.subheader("❤️ Cardiac Severity Explorer")

    c1,c2 = st.columns(2)

    if nyha_col:
        with c1:
            st.markdown("### NYHA distribution")
            ny = filtered_df[nyha_col].value_counts(dropna=False).reset_index()
            ny.columns=["NYHA","Patients"]
            fig = px.bar(ny, x="NYHA", y="Patients", title="NYHA functional class")
            st.plotly_chart(fig, use_container_width=True)

    if killip_col:
        with c2:
            st.markdown("### Killip distribution")
            ki = filtered_df[killip_col].value_counts(dropna=False).reset_index()
            ki.columns=["Killip","Patients"]
            fig = px.bar(ki, x="Killip", y="Patients", title="Killip grade")
            st.plotly_chart(fig, use_container_width=True)

    if nyha_col and killip_col and mortality28_col:
        st.markdown("### NYHA × Killip and 28-day mortality")
        temp = filtered_df[[nyha_col,killip_col,mortality28_col]].copy()
        temp["_event"] = clean_binary(temp[mortality28_col])
        pivot = temp.pivot_table(
            values="_event", index=nyha_col, columns=killip_col, aggfunc="mean"
        )*100
        fig = px.imshow(
            pivot, text_auto=".1f", aspect="auto",
            labels={"color":"28-day mortality (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    cardiac_vars = existing([
        "lvef","lvedd_mm","ea","tricuspid_valve_return_pressure",
        "tricuspid_valve_return_velocity","brain_natriuretic_peptide"
    ])
    if cardiac_vars:
        selected = st.selectbox("Cardiac measurement", cardiac_vars, format_func=pretty_col)
        outcome_name = st.selectbox("Outcome", list(OUTCOMES.keys()), key="cardiac_outcome")
        col = OUTCOMES[outcome_name]
        t = pd.DataFrame({
            "value":numeric(selected),
            "event":clean_binary(filtered_df[col])
        }).dropna()
        if len(t):
            fig = px.box(
                t, x="event", y="value", points=False,
                labels={"event":"Outcome (0 = no event, 1 = event)","value":pretty_col(selected)}
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. OUTCOMES & MORTALITY
# ============================================================
elif page == "⚠️ Outcomes & Mortality":

    st.subheader("⚠️ Outcomes & Mortality")

    outcome_name = st.selectbox("Primary outcome", list(OUTCOMES.keys()))
    outcome_col = OUTCOMES[outcome_name]

    rate = outcome_rate(filtered_df, outcome_col)
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("⚠️","Selected outcome",f"{rate:.1f}%")
    with c2: metric_card("👥","Patients",f"{len(filtered_df):,}")
    with c3:
        event_count = int(clean_binary(filtered_df[outcome_col]).sum())
        metric_card("📌","Events",f"{event_count:,}")

    st.markdown("### Outcome by clinical severity")

    if nyha_col:
        ny = filtered_df[[nyha_col,outcome_col]].copy()
        ny["_event"] = clean_binary(ny[outcome_col])
        ny = ny.dropna()
        if len(ny):
            g = ny.groupby(nyha_col)["_event"].agg(["count","mean"]).reset_index()
            g["Rate (%)"] = g["mean"]*100
            fig = px.line(
                g, x=nyha_col, y="Rate (%)", markers=True,
                title=f"{outcome_name} by NYHA class"
            )
            st.plotly_chart(fig, use_container_width=True)

    if killip_col:
        ki = filtered_df[[killip_col,outcome_col]].copy()
        ki["_event"] = clean_binary(ki[outcome_col])
        ki = ki.dropna()
        if len(ki):
            g = ki.groupby(killip_col)["_event"].agg(["count","mean"]).reset_index()
            g["Rate (%)"] = g["mean"]*100
            fig = px.line(
                g, x=killip_col, y="Rate (%)", markers=True,
                title=f"{outcome_name} by Killip grade"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Outcome profile")
    outcome_counts = clean_binary(filtered_df[outcome_col]).value_counts(dropna=False).reset_index()
    outcome_counts.columns=["Outcome","Patients"]
    outcome_counts["Outcome"] = outcome_counts["Outcome"].map(
        {0:"No event",1:"Event"}
    ).fillna("Missing")
    fig = px.pie(
        outcome_counts, names="Outcome", values="Patients",
        hole=.55, title=f"{outcome_name} distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. PRESCRIPTIVE ANALYSIS
# ============================================================
elif page == "💡 Prescriptive Analysis":

    st.subheader("💡 Prescriptive Analysis — Q1 to Q30")

    st.write(
        "This section turns the team's prescriptive notebooks into an interactive "
        "analysis library. Select a question, then choose which views to display."
    )

    q_labels = [f"{q[0]} — {q[1]}" for q in PRESCRIPTIVE_Q]
    selected_label = st.selectbox("Select analysis question", q_labels)
    qid = selected_label.split(" — ")[0]
    q = next(x for x in PRESCRIPTIVE_Q if x[0] == qid)

    title = q[1]
    question = q[2]
    variables = q[3]
    outcomes = q[4]

    available_vars = [x for x in variables if x in df.columns]
    available_outcomes = [x for x in outcomes if x in df.columns]

    st.markdown(
        f'<div class="question-box"><b>{qid}. {title}</b><br>{question}</div>',
        unsafe_allow_html=True
    )

    c1,c2,c3 = st.columns(3)
    with c1: show_main = st.checkbox("Main analysis", True)
    with c2: show_outcome = st.checkbox("Outcome comparison", True)
    with c3: show_takeaway = st.checkbox("Key takeaway", True)

    st.markdown("**Columns used:** " + ", ".join(available_vars + available_outcomes))

    if not available_vars:
        st.warning("The variables for this question are not available in the loaded dataset.")
        st.stop()

    # Main variable
    main_var = st.selectbox(
        "Variable to explore for this question",
        available_vars,
        format_func=pretty_col,
        key=f"pv_{qid}"
    )

    outcome_for_q = None
    if available_outcomes:
        outcome_for_q = st.selectbox(
            "Outcome for this view",
            available_outcomes,
            format_func=pretty_col,
            key=f"po_{qid}"
        )

    if show_main:
        st.markdown("### Main analysis")
        x = numeric(main_var)
        if x.notna().sum() >= 10:
            fig = px.histogram(
                pd.DataFrame({pretty_col(main_var):x.dropna()}),
                x=pretty_col(main_var),
                nbins=30,
                marginal="box",
                title=pretty_col(main_var)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            vc = filtered_df[main_var].astype(str).value_counts().reset_index()
            vc.columns=["Category","Patients"]
            fig = px.bar(vc, x="Category", y="Patients")
            st.plotly_chart(fig, use_container_width=True)

    if show_outcome and outcome_for_q:
        st.markdown("### Outcome comparison")
        t = pd.DataFrame({
            "x": numeric(filtered_df[main_var]),
            "event": clean_binary(filtered_df[outcome_for_q])
        }).dropna()

        if len(t) >= 10:
            try:
                t["Group"] = pd.qcut(t["x"], 4, duplicates="drop")
                g = t.groupby("Group", observed=True)["event"].agg(["count","mean"]).reset_index()
                g["Event rate (%)"] = g["mean"]*100
                fig = px.bar(
                    g, x="Group", y="Event rate (%)",
                    text="Event rate (%)",
                    title=f"{pretty_col(outcome_for_q)} across {pretty_col(main_var)} quartiles"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    g.rename(columns={"count":"Patients"})[
                        ["Group","Patients","Event rate (%)"]
                    ].round(2),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                pass
        else:
            t2 = pd.crosstab(
                filtered_df[main_var].astype(str),
                clean_binary(filtered_df[outcome_for_q]),
                normalize="index"
            )*100
            st.dataframe(t2.round(1), use_container_width=True)

    if show_takeaway:
        st.markdown("### 💡 Key takeaway")
        if outcome_for_q:
            r = outcome_rate(filtered_df, outcome_for_q)
            med = numeric(main_var).median()
            st.markdown(
                f'<div class="takeaway"><b>Selected population:</b> {len(filtered_df):,} patients. '
                f'<b>{pretty_col(outcome_for_q)}:</b> {r:.1f}%. '
                f'<b>Median {pretty_col(main_var)}:</b> {med:.2f} where numeric. '
                f'Use this view to identify patterns that can be investigated further.</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="review"><b>Prescriptive interpretation:</b> The notebook questions are presented as review priorities. '
        'The dashboard does not convert an observed association into a treatment instruction.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 7. PREDICTIVE ANALYTICS
# ============================================================
elif page == "🤖 Predictive Analytics":

    st.subheader("🤖 Predictive Analytics")
    st.write(
        "This section follows the predictive notebook: define the feature set, "
        "select the outcome, train a model, and inspect performance. "
        "It is separate from the descriptive/biomarker explorer."
    )

    pred_labels = [f"{q[0]} — {q[1]}" for q in PREDICTIVE_Q]
    pred_label = st.selectbox("Predictive question", pred_labels)
    pqid = pred_label.split(" — ")[0]
    pq = next(x for x in PREDICTIVE_Q if x[0] == pqid)

    default_features = [x for x in pq[2] if x in df.columns]
    target = pq[3] if pq[3] in df.columns else None

    st.markdown(
        f'<div class="question-box"><b>{pqid}. {pq[1]}</b><br><span class="small-muted">Target: {target or "not available"}</span></div>',
        unsafe_allow_html=True
    )

    if not target:
        st.warning("The target column for this predictive question is not available.")
        st.stop()

    selected_features = st.multiselect(
        "Model features",
        options=[c for c in df.columns if c != target],
        default=default_features,
        format_func=pretty_col
    )

    model_choice = st.selectbox(
        "Model",
        ["Logistic Regression","Random Forest","Artificial Neural Network"]
    )

    test_size = st.slider("Test set proportion", 0.20, 0.40, 0.30, 0.05)
    random_state = 42

    if len(selected_features) < 1:
        st.warning("Select at least one predictor.")
        st.stop()

    model_data = df[selected_features + [target]].copy()
    y_all = clean_binary(model_data[target])
    model_data[target] = y_all
    model_data = model_data.dropna(subset=[target])

    if model_data[target].nunique() < 2:
        st.error("The selected target does not contain both outcome classes.")
        st.stop()

    # Convert categorical variables to strings; numeric variables remain numeric.
    X_all = model_data[selected_features].copy()
    y_all = model_data[target].astype(int)

    categorical = [
        c for c in selected_features
        if X_all[c].dtype == "object" or str(X_all[c].dtype).startswith("category")
    ]
    numeric_features = [c for c in selected_features if c not in categorical]

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    preprocessor = ColumnTransformer(transformers=transformers)

    if model_choice == "Logistic Regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state
        )
    elif model_choice == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state
        )
    else:
        estimator = MLPClassifier(
            hidden_layer_sizes=(64,32),
            max_iter=600,
            early_stopping=True,
            random_state=random_state
        )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all
    )

    if st.button("▶ Run predictive model", type="primary"):
        with st.spinner("Training model..."):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test,pred)
        prec = precision_score(y_test,pred,zero_division=0)
        rec = recall_score(y_test,pred,zero_division=0)
        f1 = f1_score(y_test,pred,zero_division=0)
        auc = roc_auc_score(y_test,prob)
        pr_auc = average_precision_score(y_test,prob)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: metric_card("🎯","Accuracy",f"{acc:.3f}")
        with c2: metric_card("🔎","Precision",f"{prec:.3f}")
        with c3: metric_card("🚨","Recall",f"{rec:.3f}")
        with c4: metric_card("⚖️","F1",f"{f1:.3f}")
        with c5: metric_card("📈","ROC-AUC",f"{auc:.3f}")
        with c6: metric_card("📊","PR-AUC",f"{pr_auc:.3f}")

        col1,col2 = st.columns(2)

        with col1:
            st.markdown("### Confusion matrix")
            cm = confusion_matrix(y_test,pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                labels={"x":"Predicted","y":"Actual","color":"Patients"},
                x=["No event","Event"],
                y=["No event","Event"]
            )
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.markdown("### ROC curve")
            fpr,tpr,_ = roc_curve(y_test,prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,y=tpr,mode="lines",
                name=f"{model_choice} AUC={auc:.3f}"
            ))
            fig.add_trace(go.Scatter(
                x=[0,1],y=[0,1],mode="lines",
                name="Chance"
            ))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig,use_container_width=True)

        st.markdown(
            '<div class="research-note"><b>Model note:</b> Performance depends on the selected feature set, '
            'random split and target prevalence. A single train/test result is not a substitute for external validation.</div>',
            unsafe_allow_html=True
        )

        if model_choice in ["Logistic Regression","Random Forest"]:
            st.markdown("### Feature information")
            if model_choice == "Random Forest":
                transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
                importances = model.named_steps["model"].feature_importances_
                fi = pd.DataFrame({
                    "Feature": transformed_names,
                    "Importance": importances
                }).sort_values("Importance",ascending=False).head(20)
                fig = px.bar(
                    fi.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    title="Top model features"
                )
                st.plotly_chart(fig,use_container_width=True)
            else:
                coef = model.named_steps["model"].coef_[0]
                names = model.named_steps["preprocessor"].get_feature_names_out()
                fi = pd.DataFrame({"Feature":names,"Coefficient":coef})
                fi["Absolute coefficient"] = fi["Coefficient"].abs()
                fi = fi.sort_values("Absolute coefficient",ascending=False).head(20)
                fig = px.bar(
                    fi.sort_values("Coefficient"),
                    x="Coefficient", y="Feature", orientation="h",
                    title="Largest logistic-regression coefficients"
                )
                st.plotly_chart(fig,use_container_width=True)

    st.markdown(
        '<div class="review"><b>Clinical interpretation:</b> Predictive performance should be interpreted as model performance on the available dataset, not as a diagnosis or treatment recommendation for an individual patient.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================
elif page == "🧬 Feature Engineering":

    st.subheader("🧬 Feature Engineering Roadmap")
    st.write(
        "This page makes the transition from descriptive analysis to predictive modeling visible. "
        "The dashboard does not silently create clinical thresholds; it shows the derived features "
        "that the project can use after the underlying variables have been understood."
    )

    engineered = [
        ("NLR","neutrophil_count / lymphocyte_count","Inflammation / immune balance"),
        ("BMI category","BMI grouped into weight-status categories","Demographic / body composition"),
        ("Obesity flag","Derived from BMI category","Body composition"),
        ("Comorbidity count","Count of recorded comorbidity indicators","Disease burden"),
        ("Medication burden","Derived from total drug count","Medication intensity"),
        ("Polypharmacy flag","Derived from medication count","Medication burden"),
        ("Inflammation + albumin profile","Inflamed vs low albumin vs both vs neither","Combined biomarker profile"),
        ("Cardiac + renal profile","Abnormal cardiac and renal marker combinations","Cardiorenal burden"),
        ("Shock index","Pulse / systolic blood pressure","Hemodynamic status"),
        ("Age / severity combinations","Age stratified with NYHA/Killip/CCI","Risk stratification")
    ]

    eng_df = pd.DataFrame(engineered,columns=["Feature","How it is derived","Purpose"])
    st.dataframe(eng_df,use_container_width=True,hide_index=True)

    st.markdown("### Why this order matters")
    steps = [
        ("1","Understand the raw variables","Describe distributions, missingness and outcome rates."),
        ("2","Identify useful patterns","Use the biomarker and clinical explorers."),
        ("3","Engineer features","Create reproducible derived variables."),
        ("4","Build predictive models","Compare feature sets and model families."),
        ("5","Validate","Use cross-validation / external validation before clinical use.")
    ]
    for n,title,desc in steps:
        st.markdown(
            f'<div class="section-card"><b>{n}. {title}</b><br><span class="small-muted">{desc}</span></div>',
            unsafe_allow_html=True
        )

# ============================================================
# 9. DATA QUALITY
# ============================================================
elif page == "🧹 Data Quality":

    st.subheader("🧹 Data Quality")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("👥","Rows",f"{len(df):,}")
    with c2: metric_card("🧱","Columns",f"{df.shape[1]:,}")
    with c3: metric_card("🆔","Unique patients",f"{df['inpatient_number'].nunique():,}" if "inpatient_number" in df.columns else "N/A")
    with c4: metric_card("🔁","Duplicate rows",f"{df.duplicated().sum():,}")

    quality = pd.DataFrame({
        "Column":df.columns,
        "Missing":df.isna().sum().values,
        "Missing (%)":(df.isna().mean().values*100).round(2),
        "Unique":df.nunique(dropna=True).values,
        "Type":[str(x) for x in df.dtypes]
    }).sort_values("Missing (%)",ascending=False)

    st.markdown("### Missingness by variable")
    fig = px.bar(
        quality.head(30).sort_values("Missing (%)"),
        x="Missing (%)", y="Column", orientation="h",
        title="Top variables by missingness"
    )
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Data-quality table")
    st.dataframe(quality,use_container_width=True,hide_index=True)

    st.download_button(
        "📥 Download data-quality report",
        quality.to_csv(index=False).encode("utf-8"),
        "heart_failure_data_quality.csv",
        "text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "HeartCare AI • Research and analytical dashboard • "
    "Observed associations are not causal conclusions. Predictive models require validation before clinical deployment."
)
