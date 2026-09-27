import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

st.set_page_config(
    page_title="HeartCare Clinical Explorer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
<style>
.stApp { background:#F4F9FB; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#073B4C,#0B5D6B 55%,#087F5B); }
section[data-testid="stSidebar"] * { color:white !important; }
.main-title { font-size:34px; font-weight:700; color:#073B4C; }
.subtitle { font-size:16px; color:#52727D; margin-bottom:20px; }
.hospital-header { background:linear-gradient(90deg,#073B4C,#087F5B); padding:24px 28px; border-radius:16px; color:white; margin-bottom:24px; box-shadow:0 5px 15px rgba(0,0,0,.08); }
.hospital-header h1 { margin:0; font-size:30px; }
.hospital-header p { margin:5px 0 0; opacity:.92; }
.info-box { background:#EAF5F8; border-left:5px solid #087F9B; padding:15px; border-radius:9px; margin:10px 0 18px; }
.metric-card { background:white; padding:18px; border-radius:15px; border-left:5px solid #087F5B; box-shadow:0 3px 12px rgba(0,0,0,.06); min-height:105px; }
.metric-icon { font-size:26px; }.metric-title{color:#637B83;font-size:13px;font-weight:600}.metric-value{color:#073B4C;font-size:27px;font-weight:700}
.section-card { background:white; padding:20px; border-radius:15px; box-shadow:0 3px 12px rgba(0,0,0,.05); margin-bottom:18px; }
.risk-high { background:#FFF1F0; border-left:6px solid #D9534F; padding:18px; border-radius:10px; }
.risk-low { background:#EAF8F1; border-left:6px solid #087F5B; padding:18px; border-radius:10px; }
.small-note { color:#637B83; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data loading
# ----------------------------
DATA_FILE = Path(__file__).parent / "Cardiac_Cleaned_Data.xlsb"

@st.cache_data(show_spinner="Loading cleaned cardiac dataset…")
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE.name}. Place it beside this Python file.")
    return pd.read_excel(DATA_FILE, engine="pyxlsb")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load Cardiac_Cleaned_Data.xlsb: {e}")
    st.stop()

# Clean column labels without changing the underlying data values.
df.columns = [str(c).strip() for c in df.columns]

# ----------------------------
# Flexible column detection
# ----------------------------
def norm(x):
    return ''.join(ch for ch in str(x).lower() if ch.isalnum())

NORMALIZED = {norm(c): c for c in df.columns}

def find_col(aliases, contains=True):
    # exact normalized match first
    for a in aliases:
        na = norm(a)
        if na in NORMALIZED:
            return NORMALIZED[na]
    if contains:
        for a in aliases:
            na = norm(a)
            if len(na) < 4:
                continue
            for nc, original in NORMALIZED.items():
                if na in nc or nc in na:
                    return original
    return None

def numeric_col(aliases):
    c = find_col(aliases)
    return c

patient_id = find_col(["inpatient_number", "patient_id", "patientid", "id"], contains=False)
gender_col = find_col(["gender", "sex"])
age_col = find_col(["age", "age_years", "ageyears"])
agecat_col = find_col(["agecat", "age_category", "agegroup"])
bmi_col = find_col(["bmi", "body_mass_index"])
weight_col = find_col(["weight", "body_weight"])
height_col = find_col(["height", "body_height"])
occupation_col = find_col(["occupation"])
nyha_col = find_col(["nyha_cardiac_function_classification", "nyha_cardiac_function", "nyha_class", "nyha"])
killip_col = find_col(["killip_grade", "killip_class", "killip"])
hf_type_col = find_col(["type_of_heart_failure", "heart_failure_type"])
lvef_col = find_col(["lvef", "left_ventricular_ejection_fraction"])
lvedd_col = find_col(["lvedd_mm", "left_ventricular_end_diastolic_diameter_lv", "lvedd"])

# Biomarker / clinical columns
aliases = {
    "hs-CRP": ["hs_crp", "hs-crp", "hsCRP", "high_sensitivity_crp", "crp"],
    "WBC": ["wbc", "white_blood_cell_count", "white_blood_cells"],
    "NLR": ["nlr", "neutrophil_lymphocyte_ratio"],
    "Albumin": ["albumin"],
    "BNP": ["brain_natriuretic_peptide", "bnp"],
    "Troponin": ["high_sensitivity_troponin", "hs_troponin", "troponin", "troponini"],
    "CK-MB": ["ck_mb", "creatine_kinase_mb", "ckmb"],
    "Myoglobin": ["myoglobin"],
    "Creatinine": ["creatinine_enzymatic_method", "creatinine", "creatinine_enzymatic"],
    "eGFR": ["egfr", "estimated_glomerular_filtration_rate"],
    "Urea": ["urea", "blood_urea_nitrogen", "bun"],
    "Cystatin": ["cystatin", "cystatin_c"],
    "D-dimer": ["d_dimer", "d-dimer", "ddimer"],
    "INR": ["inr", "international_normalized_ratio"],
    "Sodium": ["sodium", "na"],
    "Potassium": ["potassium", "k"],
    "Hemoglobin": ["hemoglobin", "hb"],
    "RDW": ["rdw"],
    "Lactate": ["lactate"],
    "pH": ["ph", "blood_ph"],
    "Oxygen Saturation": ["oxygen_saturation", "spo2", "oxygen_sat"],
    "Total Bilirubin": ["total_bilirubin", "bilirubin"],
    "AST/ALT Ratio": ["ast_alt_ratio", "ast_alt", "astalt_ratio"],
    "Respiratory Support": ["respiratory_support"],
    "CCI": ["cci", "charlson_comorbidity_index", "charlson_comorbidity_score"],
    "Admission Way": ["admission_way", "admission_type", "admission_method"],
    "Discharge Destination": ["destinationdischarge", "discharge_destination", "destination_discharge"],
    "LOS": ["dischargeday", "length_of_stay", "los", "hospital_stay_days"],
}
clinical = {label: find_col(names) for label, names in aliases.items()}

# Outcomes
outcome_col = find_col(["outcome_during_hospitalization"], contains=False)
if outcome_col:
    outcome_text = df[outcome_col].astype(str).str.strip().str.lower()
    df["In-Hospital Mortality"] = outcome_text.eq("dead").astype(int)
    mortality_col = "In-Hospital Mortality"
else:
    mortality_col = find_col(["in_hospital_mortality", "hospital_mortality", "mortality", "death"])

mortality28_col = find_col(["death_within_28_days", "death_28_days", "mortality_28d", "28_day_mortality"])
mortality3_col = find_col(["death_within_3_months", "death_3_months", "mortality_3m"])
mortality6_col = find_col(["death_within_6_months", "death_6_months", "mortality_6m"])
readmit28_col = find_col(["re_admission_within_28_days", "readmission_within_28_days", "readmission_28d"])
readmit3_col = find_col(["re_admission_within_3_months", "readmission_within_3_months", "readmission_3m"])
readmit6_col = find_col(["re_admission_within_6_months", "readmission_within_6_months", "readmission_6m"])
ed6_col = find_col(["return_to_emergency_department_within_6_months", "ed_return_6m", "emergency_department_return"])
readmit_days_col = find_col(["readmission_time_days_from_admission", "readmission_time_days"])

# Convert binary outcomes safely when needed.
def as_binary(series):
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        return vals.where(vals.isin([0,1]))
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes":1,"y":1,"true":1,"1":1,"dead":1,"death":1,"died":1,
        "no":0,"n":0,"false":0,"0":0,"alive":0,"survived":0
    }
    return s.map(mapping)

# ----------------------------
# Feature engineering for dashboard exploration
# ----------------------------
if bmi_col:
    bmi_num = pd.to_numeric(df[bmi_col], errors="coerce")
    df["BMI Category"] = pd.cut(
        bmi_num, bins=[-np.inf,18.5,25,30,np.inf],
        labels=["Underweight","Normal","Overweight","Obese"]
    )

if clinical.get("CCI"):
    cci_num = pd.to_numeric(df[clinical["CCI"]], errors="coerce")
    df["CCI Group"] = pd.cut(
        cci_num, bins=[-np.inf,2,4,np.inf],
        labels=["0–2","3–4","5+"]
    )

# Inflammation + albumin flags follow the project's stated thresholds.
crp = clinical.get("hs-CRP")
wbc = clinical.get("WBC")
nlr = clinical.get("NLR")
albumin = clinical.get("Albumin")
if crp or wbc or nlr:
    flags = []
    if crp: flags.append(pd.to_numeric(df[crp], errors="coerce") > 5)
    if wbc: flags.append(pd.to_numeric(df[wbc], errors="coerce") > 10)
    if nlr: flags.append(pd.to_numeric(df[nlr], errors="coerce") > 6)
    df["Inflammation Flag"] = pd.concat(flags, axis=1).any(axis=1).astype("int8")
else:
    df["Inflammation Flag"] = 0
if albumin:
    df["Low Albumin Flag"] = (pd.to_numeric(df[albumin], errors="coerce") < 35).astype("int8")
else:
    df["Low Albumin Flag"] = 0

df["Inflammation + Albumin Group"] = np.select(
    [
        (df["Inflammation Flag"]==0) & (df["Low Albumin Flag"]==0),
        (df["Inflammation Flag"]==1) & (df["Low Albumin Flag"]==0),
        (df["Inflammation Flag"]==0) & (df["Low Albumin Flag"]==1),
        (df["Inflammation Flag"]==1) & (df["Low Albumin Flag"]==1),
    ],
    ["Neither", "Inflamed only", "Low albumin only", "Both"],
    default="Unknown"
).astype(str)

# ----------------------------
# Safe plotting helpers
# ----------------------------
def show_plot(fig, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.warning("This chart could not be rendered from the available values. The table below contains the same summary.")

def numeric_series(col):
    return pd.to_numeric(df[col], errors="coerce") if col else pd.Series(dtype=float)

def outcome_rate(data, outcome):
    if outcome not in data.columns:
        return np.nan
    return as_binary(data[outcome]).mean() * 100

def safe_group_rate(data, group_col, outcome):
    if not group_col or not outcome or group_col not in data.columns or outcome not in data.columns:
        return pd.DataFrame()
    tmp = pd.DataFrame({"Group": data[group_col].astype(str), "Outcome": as_binary(data[outcome])}).dropna()
    if tmp.empty: return tmp
    out = tmp.groupby("Group", dropna=False)["Outcome"].agg(["count","mean"]).reset_index()
    out["Rate (%)"] = out["mean"] * 100
    return out[["Group","count","Rate (%)"]]

def make_hist(data, col, title):
    s = pd.to_numeric(data[col], errors="coerce")
    plot = pd.DataFrame({"Value":s}).dropna()
    if plot.empty:
        st.info(f"No numeric values available for {title}.")
        return
    fig = px.histogram(plot, x="Value", nbins=30, marginal="box", title=title)
    show_plot(fig, 430)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:12px'><div style='font-size:48px'>❤️</div><h2>HeartCare AI</h2><p>Clinical Analytics Explorer</p></div>", unsafe_allow_html=True)
    st.divider()
    pages = [
        "🏠 Introduction", "🏥 Overview", "🧹 Data Cleaning", "🧬 Feature Engineering",
        "📈 Descriptive Analysis", "💡 Prescriptive Analysis", "🤖 Predictive Analytics",
        "📊 Model Performance", "👤 Patient Profile", "⭐ Insights & Key Takeaways", "🏁 Conclusion"
    ]
    page = st.radio("NAVIGATION", pages)
    st.divider()
    st.caption(f"Rows loaded: {len(df):,}")
    st.caption("For research and analytical use; not a clinical diagnosis.")

st.markdown("""
<div class="hospital-header">
<h1>❤️ Heart Failure Clinical Analytics</h1>
<p>Demographics • Cardiac Severity • Biomarkers • Outcomes • Predictive Modeling • Clinical Exploration</p>
</div>
""", unsafe_allow_html=True)

def metric_card(icon,title,value):
    st.markdown(f"<div class='metric-card'><div class='metric-icon'>{icon}</div><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

# ----------------------------
# Introduction
# ----------------------------
if page == "🏠 Introduction":
    st.markdown("<div class='main-title'>Introduction</div><div class='subtitle'>Heart-failure clinical analytics project: from cleaned patient data to descriptive, prescriptive and predictive analysis.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'><b>Project question</b><br>
    Can demographic, clinical, cardiac, laboratory, nutritional and responsiveness characteristics be combined in an Artificial Neural Network model to predict mortality among patients with heart failure?</div>
    """, unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("🧹","Data layer","Cleaned patient-level dataset")
    with c2: metric_card("📊","Analytics layer","Descriptive + prescriptive")
    with c3: metric_card("🤖","Prediction layer","ANN + comparison models")
    st.subheader("How the project flows")
    st.markdown("**Raw tables → Data cleaning → One patient-level dataset → Feature engineering → Descriptive analysis → Prescriptive questions → Predictive models → Dashboard insights**")
    st.subheader("Outcomes examined")
    outcome_items=[]
    if mortality_col: outcome_items.append("In-hospital mortality")
    if mortality28_col: outcome_items.append("28-day mortality")
    if mortality3_col: outcome_items.append("3-month mortality")
    if mortality6_col: outcome_items.append("6-month mortality")
    if readmit28_col: outcome_items.append("28-day readmission")
    if readmit3_col: outcome_items.append("3-month readmission")
    if readmit6_col: outcome_items.append("6-month readmission")
    if ed6_col: outcome_items.append("6-month ED return")
    st.write(" • ".join(outcome_items) if outcome_items else "Outcome columns were not detected automatically.")
    st.subheader("How to use this dashboard")
    st.markdown("Use the sidebar to move from the project introduction and data-quality evidence through clinical patterns, the 30 prescriptive questions, predictive modeling, model performance and patient-level exploration.")

# ----------------------------
# Overview
# ----------------------------
elif page == "🏥 Overview":
    st.markdown("<div class='main-title'>Hospital Overview</div><div class='subtitle'>Population, outcomes and available clinical information.</div>", unsafe_allow_html=True)
    mort_rate = outcome_rate(df,mortality_col) if mortality_col else np.nan
    avg_age = numeric_series(age_col).mean() if age_col else np.nan
    avg_bmi = numeric_series(bmi_col).mean() if bmi_col else np.nan
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("👥","Patients",f"{len(df):,}")
    with c2: metric_card("⚠️","In-hospital mortality",f"{mort_rate:.1f}%" if pd.notna(mort_rate) else "N/A")
    with c3: metric_card("🎂","Mean age",f"{avg_age:.1f}" if pd.notna(avg_age) else (f"{df[agecat_col].notna().sum():,} age groups" if agecat_col else "N/A"))
    with c4: metric_card("⚖️","Mean BMI",f"{avg_bmi:.1f}" if pd.notna(avg_bmi) else "N/A")
    st.markdown("<div class='info-box'><b>Clinical view:</b> This page gives a quick population-level starting point. Use the other pages for severity, biomarker, outcome and model details.</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        if gender_col:
            g=df[gender_col].astype(str).value_counts().reset_index(); g.columns=["Gender","Patients"]
            show_plot(px.bar(g,x="Gender",y="Patients",text="Patients",title="Patient distribution by gender"),400)
    with c2:
        if agecat_col:
            a=df[agecat_col].astype(str).value_counts().reset_index(); a.columns=["Age group","Patients"]
            show_plot(px.bar(a,x="Age group",y="Patients",text="Patients",title="Patient distribution by age group"),400)
        elif age_col:
            make_hist(df,age_col,"Age distribution")
    if mortality_col:
        st.subheader("Outcome distribution")
        o=as_binary(df[mortality_col]).value_counts(dropna=False).reset_index(); o.columns=["Outcome","Patients"]; o["Outcome"]=o["Outcome"].map({0:"No in-hospital death",1:"In-hospital death"}).fillna("Missing")
        show_plot(px.pie(o,names="Outcome",values="Patients",hole=.55),400)

# ----------------------------
# Data Cleaning
# ----------------------------
elif page == "🧹 Data Cleaning":
    st.markdown("<div class='main-title'>Data Cleaning & Quality Review</div><div class='subtitle'>Evidence about the final patient-level dataset used by the dashboard.</div>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: metric_card("📄","Rows",f"{len(df):,}")
    with c2: metric_card("🧾","Columns",f"{df.shape[1]:,}")
    with c3: metric_card("🆔","Unique patients",f"{df[patient_id].nunique():,}" if patient_id else "Not detected")
    with c4: metric_card("♻️","Duplicate rows",f"{df.duplicated().sum():,}")
    st.subheader("Column inventory")
    inventory=pd.DataFrame({"Column":df.columns,"Data type":[str(df[c].dtype) for c in df.columns],"Missing":df.isna().sum().values,"Missing %":(df.isna().mean()*100).round(1).values})
    st.dataframe(inventory,use_container_width=True,hide_index=True)
    st.subheader("Highest missingness")
    miss=inventory.sort_values("Missing %",ascending=False).head(15)
    show_plot(px.bar(miss.sort_values("Missing %"),x="Missing %",y="Column",orientation="h",title="Top columns by missingness"),500)
    st.markdown("<div class='info-box'><b>Interpretation:</b> Missing clinical measurements are shown as missing rather than automatically converted to zero. This matters because an unmeasured biomarker and a measured value of zero have different clinical meanings.</div>",unsafe_allow_html=True)

# ----------------------------
# Feature Engineering
# ----------------------------
elif page == "🧬 Feature Engineering":
    st.markdown("<div class='main-title'>Feature Engineering</div><div class='subtitle'>Derived variables used to make clinical patterns easier to explore.</div>", unsafe_allow_html=True)
    derived=[c for c in ["BMI Category","CCI Group","Inflammation Flag","Low Albumin Flag","Inflammation + Albumin Group"] if c in df.columns]
    if not derived: st.info("No derived features were created from the detected columns.")
    else:
        st.dataframe(pd.DataFrame({"Derived feature":derived,"Description":[
            "BMI grouped as underweight, normal, overweight and obese.",
            "CCI grouped as 0–2, 3–4 and 5+.",
            "Flag when hs-CRP > 5, WBC > 10 or NLR > 6 where available.",
            "Flag when albumin < 35 where available.",
            "Combines inflammation and low-albumin flags."
        ][:len(derived)]}),use_container_width=True,hide_index=True)
        for c in derived:
            counts=df[c].astype(str).value_counts().reset_index(); counts.columns=[c,"Patients"]
            show_plot(px.bar(counts,x=c,y="Patients",text="Patients",title=c),350)

# ----------------------------
# Patient Profile
# ----------------------------
elif page == "👤 Patient Profile":
    st.markdown("<div class='main-title'>Patient Explorer</div><div class='subtitle'>Filter the patient-level dataset and inspect a selected record.</div>", unsafe_allow_html=True)
    filtered=df.copy()
    f1,f2,f3,f4=st.columns(4)
    if gender_col:
        vals=sorted(df[gender_col].dropna().astype(str).unique().tolist()); sel= f1.multiselect("Gender",vals); 
        if sel: filtered=filtered[filtered[gender_col].astype(str).isin(sel)]
    if nyha_col:
        vals=sorted(df[nyha_col].dropna().astype(str).unique().tolist()); sel=f2.multiselect("NYHA",vals)
        if sel: filtered=filtered[filtered[nyha_col].astype(str).isin(sel)]
    if killip_col:
        vals=sorted(df[killip_col].dropna().astype(str).unique().tolist()); sel=f3.multiselect("Killip",vals)
        if sel: filtered=filtered[filtered[killip_col].astype(str).isin(sel)]
    if "BMI Category" in df.columns:
        vals=sorted(df["BMI Category"].dropna().astype(str).unique().tolist()); sel=f4.multiselect("BMI category",vals)
        if sel: filtered=filtered[filtered["BMI Category"].astype(str).isin(sel)]
    st.write(f"**Patients matching filters: {len(filtered):,}**")
    if patient_id and len(filtered):
        ids=filtered[patient_id].dropna().tolist(); selected_id=st.selectbox("Select patient ID",ids)
        row=filtered[filtered[patient_id]==selected_id].iloc[0]
        st.subheader("Patient profile")
        cols=st.columns(4); profile_items=[]
        for label,col in [("Gender",gender_col),("Age",age_col or agecat_col),("BMI",bmi_col),("NYHA",nyha_col),("Killip",killip_col),("HF type",hf_type_col),("LVEF",lvef_col),("LVEDD",lvedd_col),("CCI",clinical.get("CCI")),("Outcome",outcome_col)]:
            if col and col in row.index: profile_items.append((label,row[col]))
        for i,(label,val) in enumerate(profile_items):
            with cols[i%4]: st.metric(label,str(val) if pd.notna(val) else "Missing")
        st.dataframe(pd.DataFrame({"Variable":[x[0] for x in profile_items],"Value":[x[1] for x in profile_items]}),use_container_width=True,hide_index=True)
    else:
        st.dataframe(filtered.head(100),use_container_width=True,hide_index=True)

# ----------------------------
# Cardiac Risk
# ----------------------------
elif page == "❤️ Cardiac Risk":
    st.markdown("<div class='main-title'>Cardiac Severity</div><div class='subtitle'>NYHA, Killip and structural/functional cardiac characteristics.</div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        if nyha_col:
            tab=safe_group_rate(df,nyha_col,mortality_col) if mortality_col else pd.DataFrame()
            if not tab.empty: show_plot(px.bar(tab,x="Group",y="Rate (%)",text="Rate (%)",title="In-hospital mortality by NYHA"),420); st.dataframe(tab,use_container_width=True,hide_index=True)
            else: st.info("NYHA detected, but the selected mortality outcome is unavailable.")
        else: st.warning("NYHA column not detected in the cleaned file.")
    with c2:
        if killip_col:
            tab=safe_group_rate(df,killip_col,mortality_col) if mortality_col else pd.DataFrame()
            if not tab.empty: show_plot(px.bar(tab,x="Group",y="Rate (%)",text="Rate (%)",title="In-hospital mortality by Killip"),420); st.dataframe(tab,use_container_width=True,hide_index=True)
            else: st.info("Killip detected, but the selected mortality outcome is unavailable.")
        else: st.warning("Killip column not detected in the cleaned file.")
    if nyha_col and killip_col and mortality_col:
        tmp=pd.DataFrame({"NYHA":df[nyha_col].astype(str),"Killip":df[killip_col].astype(str),"Death":as_binary(df[mortality_col])}).dropna()
        pv=tmp.pivot_table(values="Death",index="NYHA",columns="Killip",aggfunc="mean")*100
        pv=pv.astype(float)
        fig=px.imshow(pv,text_auto=".1f",aspect="auto",labels={"color":"Mortality (%)"},title="NYHA × Killip mortality (%)")
        show_plot(fig,450)
    c1,c2=st.columns(2)
    for col,title,target in [(lvef_col,"LVEF distribution",mortality_col),(lvedd_col,"LVEDD distribution",mortality_col)]:
        if col:
            with (c1 if title.startswith("LVEF") else c2):
                make_hist(df,col,title)

# ----------------------------
# Biomarkers
# ----------------------------
elif page == "🧪 Biomarkers":
    st.markdown("<div class='main-title'>Biomarker & Nutrition Explorer</div><div class='subtitle'>Choose from the biomarkers actually detected in the cleaned dataset.</div>", unsafe_allow_html=True)
    biomarker_cols=[(label,col) for label,col in clinical.items() if col and label not in ["Respiratory Support","CCI","Admission Way","Discharge Destination","LOS"]]
    if not biomarker_cols:
        st.warning("No biomarker columns were detected.")
    else:
        labels=[x[0] for x in biomarker_cols]; selected_label=st.selectbox("Select biomarker",labels); selected=dict(biomarker_cols)[selected_label]
        c1,c2=st.columns(2)
        with c1: make_hist(df,selected,f"{selected_label} distribution")
        with c2:
            if mortality_col:
                tmp=pd.DataFrame({"Value":pd.to_numeric(df[selected],errors="coerce"),"Outcome":as_binary(df[mortality_col])}).dropna()
                if len(tmp)>=10:
                    try:
                        tmp["Biomarker Group"]=pd.qcut(tmp["Value"],q=3,duplicates="drop").astype(str)
                        res=tmp.groupby("Biomarker Group",observed=False)["Outcome"].mean().reset_index(); res["Mortality (%)"]=res["Outcome"]*100
                        show_plot(px.bar(res,x="Biomarker Group",y="Mortality (%)",text="Mortality (%)",title=f"{selected_label} vs in-hospital mortality"),420)
                        st.dataframe(res[["Biomarker Group","Mortality (%)"]],use_container_width=True,hide_index=True)
                    except Exception as e:
                        st.info("Not enough unique values to form biomarker groups.")
        summary=pd.DataFrame({"Biomarker":[x[0] for x in biomarker_cols],"Column":[x[1] for x in biomarker_cols],"Available values":[df[x[1]].notna().sum() for x in biomarker_cols],"Missing %":[round(df[x[1]].isna().mean()*100,1) for x in biomarker_cols]})
        st.subheader("Available biomarkers")
        st.dataframe(summary,use_container_width=True,hide_index=True)

# ----------------------------
# Descriptive
# ----------------------------
elif page == "📈 Descriptive Analysis":
    st.markdown("<div class='main-title'>Descriptive Analysis</div><div class='subtitle'>What does the patient population look like?</div>", unsafe_allow_html=True)
    numeric_candidates=[]
    for label,col in clinical.items():
        if col and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col],errors="coerce")):
            numeric_candidates.append((label,col))
    selected_label=st.selectbox("Variable to describe",[x[0] for x in numeric_candidates] or ["No numeric variables detected"])
    if numeric_candidates:
        col=dict(numeric_candidates)[selected_label]; s=pd.to_numeric(df[col],errors="coerce")
        c1,c2,c3,c4=st.columns(4)
        with c1: metric_card("N","Available",f"{s.notna().sum():,}")
        with c2: metric_card("μ","Mean",f"{s.mean():.2f}" if s.notna().any() else "N/A")
        with c3: metric_card("M","Median",f"{s.median():.2f}" if s.notna().any() else "N/A")
        with c4: metric_card("%","Missing",f"{s.isna().mean()*100:.1f}%")
        make_hist(df,col,f"Distribution: {selected_label}")
    st.subheader("Outcome rates")
    outcomes=[("In-hospital mortality",mortality_col),("28-day mortality",mortality28_col),("3-month mortality",mortality3_col),("6-month mortality",mortality6_col),("28-day readmission",readmit28_col),("3-month readmission",readmit3_col),("6-month readmission",readmit6_col),("6-month ED return",ed6_col)]
    rates=[(label,outcome_rate(df,col)) for label,col in outcomes if col]
    if rates: show_plot(px.bar(pd.DataFrame(rates,columns=["Outcome","Rate (%)"]),x="Outcome",y="Rate (%)",text="Rate (%)",title="Available outcome rates"),430)

# ----------------------------
# Prescriptive Q1-Q30
# ----------------------------
elif page == "💡 Prescriptive Analysis":
    st.markdown("<div class='main-title'>Prescriptive Analysis: Q1–Q30</div><div class='subtitle'>The project questions translated into dashboard review points. These are association-based analytical flags, not treatment orders.</div>", unsafe_allow_html=True)
    questions=[
        "Q1. Does NYHA severity relate to 28-day, 3-month and 6-month outcomes?",
        "Q2. Does Killip grade relate to in-hospital and 28-day mortality?",
        "Q3. Do LVEDD and E/A patterns relate to 6-month outcomes?",
        "Q4. Do left/right/both-sided heart-failure patterns relate to pressure, BNP and outcomes?",
        "Q5. Do prior MI/PVD patterns relate to cardiac injury and medication use?",
        "Q6. Is BNP associated with adverse outcomes?",
        "Q7. Are troponin, CK-MB and myoglobin associated with mortality?",
        "Q8. Do high inflammation markers together with low albumin identify higher-risk groups?",
        "Q9. Do hemoglobin and RDW patterns differ across NYHA/outcomes?",
        "Q10. Are sodium/potassium ranges linked with outcomes and diuretic use?",
        "Q11. Do D-dimer and INR patterns differ with mortality and anticoagulant use?",
        "Q12. Do liver-congestion markers and low albumin relate to pressure and outcomes?",
        "Q13. Do lactate, pH and oxygen saturation relate to in-hospital death?",
        "Q14. Do kidney-function markers or CKD stage relate to outcomes?",
        "Q15. Does combined cardiac + renal burden identify a higher-risk group?",
        "Q16. Do acute kidney failure and CKD show different outcome patterns?",
        "Q17. Does CCI group relate to mortality/readmission?",
        "Q18. Does high comorbidity combined with high medication burden identify a review group?",
        "Q19. Does COPD/type II respiratory failure relate to CO2, oxygen and readmission?",
        "Q20. Does ventilation/respiratory support differ by NYHA, BNP, troponin and outcomes?",
        "Q21. Does neurological disease relate to non-home discharge or death?",
        "Q22. How do guideline medication indicators vary across patients?",
        "Q23. Do inotropes identify an advanced-heart-failure profile?",
        "Q24. Does low blood pressure or high shock index relate to death?",
        "Q25. Are underweight patients different in outcome, and does low albumin co-occur?",
        "Q26. Which age/gender patterns remain different within severity/comorbidity groups?",
        "Q27. Do emergency admissions differ in LOS and outcomes from planned admissions?",
        "Q28. Are very short stays associated with earlier readmission?",
        "Q29. Which discharge destinations have different readmission/ED-return rates?",
        "Q30. Who are frequent returners, and what clinical profile do they share?",
    ]
    q=st.selectbox("Select a project question",questions)
    st.markdown(f"### {q}")
    st.markdown("<div class='info-box'><b>Dashboard interpretation:</b> Use the displayed group differences as descriptive evidence for clinical review. Associations do not by themselves establish causation or prescribe treatment.</div>",unsafe_allow_html=True)
    # Connect common questions to live analyses.
    if q.startswith("Q1") and nyha_col:
        for label,col in [("28-day mortality",mortality28_col),("3-month mortality",mortality3_col),("6-month mortality",mortality6_col),("6-month readmission",readmit6_col)]:
            if col:
                tab=safe_group_rate(df,nyha_col,col); st.subheader(label); st.dataframe(tab,use_container_width=True,hide_index=True)
    elif q.startswith("Q2") and killip_col and mortality_col:
        st.dataframe(safe_group_rate(df,killip_col,mortality_col),use_container_width=True,hide_index=True)
    elif q.startswith("Q8"):
        g=df["Inflammation + Albumin Group"].value_counts().reset_index(); g.columns=["Group","Patients"]; show_plot(px.bar(g,x="Group",y="Patients",text="Patients",title="Inflammation + albumin groups"),400)
        if mortality_col:
            tab=safe_group_rate(df,"Inflammation + Albumin Group",mortality_col); st.dataframe(tab,use_container_width=True,hide_index=True)
    elif q.startswith("Q17") and "CCI Group" in df.columns and mortality_col:
        st.dataframe(safe_group_rate(df,"CCI Group",mortality_col),use_container_width=True,hide_index=True)
    elif q.startswith("Q25") and "BMI Category" in df.columns:
        tab=safe_group_rate(df,"BMI Category",mortality_col) if mortality_col else pd.DataFrame(); st.dataframe(tab,use_container_width=True,hide_index=True)
        if albumin:
            ab=df.groupby("BMI Category",observed=False)[albumin].mean().reset_index(); st.dataframe(ab,use_container_width=True,hide_index=True)
    elif q.startswith("Q27") and clinical.get("Admission Way") and clinical.get("LOS"):
        adm=clinical["Admission Way"]; los=clinical["LOS"]; tmp=pd.DataFrame({"Admission":df[adm].astype(str),"LOS":pd.to_numeric(df[los],errors="coerce")}).dropna(); st.dataframe(tmp.groupby("Admission")["LOS"].agg(["count","mean","median"]).reset_index(),use_container_width=True,hide_index=True)
    else:
        st.info("This question is included in the dashboard structure. A live calculation is shown when the required source columns are detected in the cleaned file.")

# ----------------------------
# Modeling helpers
# ----------------------------
def build_model_data(target, max_features=24):
    if not target: return None,None,None,[]
    candidates=[]
    priority=[age_col,agecat_col,gender_col,bmi_col,nyha_col,killip_col,hf_type_col,lvef_col,lvedd_col]+[clinical[k] for k in ["hs-CRP","WBC","NLR","Albumin","BNP","Troponin","Creatinine","eGFR","Urea","D-dimer","Sodium","Potassium","Hemoglobin","Lactate","Oxygen Saturation","CCI"] if clinical.get(k)]
    for c in priority:
        if c and c not in candidates and c != target and c in df.columns: candidates.append(c)
    candidates=candidates[:max_features]
    work=df[candidates+[target]].copy()
    y=as_binary(work[target])
    work=work.loc[y.notna()].copy(); y=as_binary(work[target]).astype(int)
    X=work[candidates].copy()
    # Drop columns with no usable values.
    good=[c for c in X.columns if X[c].notna().any()]
    X=X[good]
    return X,y,work,good

def model_pipeline(X, kind):
    cats=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums=[c for c in X.columns if c not in cats]
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)
    ],remainder="drop")
    if kind=="Logistic Regression":
        clf=LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42)
    elif kind=="Random Forest":
        clf=RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1)
    else:
        clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=600,early_stopping=True,random_state=42)
    return Pipeline([("prep",pre),("model",clf)])

def run_models(target):
    X,y,work,features=build_model_data(target)
    if X is None or len(features)<2 or y.nunique()<2 or len(y)<30: return None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    rows=[]; artifacts={}
    for name in ["Logistic Regression","Random Forest","ANN"]:
        model=model_pipeline(X,name); model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),"F1":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,prob),"PR-AUC":average_precision_score(yte,prob)})
        artifacts[name]=(model,pred,prob)
    return pd.DataFrame(rows),artifacts,(Xte,yte),features

# ----------------------------
# Predictive Analytics
# ----------------------------
if page == "🤖 Predictive Analytics":
    st.markdown("<div class='main-title'>Predictive Analytics</div><div class='subtitle'>Interactive mortality prediction using the same patient-level dataset and a preprocessing pipeline that handles numeric and categorical variables.</div>", unsafe_allow_html=True)
    target_options=[("In-hospital mortality",mortality_col),("28-day mortality",mortality28_col),("6-month mortality",mortality6_col)]
    target_label=st.selectbox("Prediction outcome",[x[0] for x in target_options if x[1]])
    target=dict((x[0],x[1]) for x in target_options if x[1])[target_label]
    result=run_models(target)
    if result is None:
        st.warning("Not enough usable outcome/feature data was detected to train the models.")
    else:
        metrics,artifacts,(Xte,yte),features=result
        st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),use_container_width=True,hide_index=True)
        st.subheader("Patient prediction")
        # ANN interactive form
        ann=artifacts["ANN"][0]
        input_data={}
        cols=st.columns(3)
        for i,c in enumerate(features):
            s=df[c]
            with cols[i%3]:
                if not pd.api.types.is_numeric_dtype(s):
                    vals=sorted(s.dropna().astype(str).unique().tolist())
                    if vals: input_data[c]=st.selectbox(c.replace("_"," ").title(),vals,key=f"pred_{c}")
                else:
                    sn=pd.to_numeric(s,errors="coerce").dropna()
                    if not sn.empty: input_data[c]=st.number_input(c.replace("_"," ").title(),value=float(sn.median()),min_value=float(sn.min()),max_value=float(sn.max()),key=f"pred_{c}")
        if st.button("❤️ Calculate ANN Mortality Probability",type="primary",use_container_width=True):
            patient=pd.DataFrame([input_data],columns=features)
            prob=float(ann.predict_proba(patient)[0,1]); pred=int(ann.predict(patient)[0])
            c1,c2=st.columns(2)
            with c1: metric_card("🎯","Predicted probability",f"{prob*100:.1f}%"); st.progress(prob)
            with c2:
                st.markdown(("<div class='risk-high'><h3>⚠️ Model class: mortality</h3>Research-model classification only.</div>" if pred else "<div class='risk-low'><h3>✓ Model class: non-mortality</h3>Research-model classification only.</div>"),unsafe_allow_html=True)
            st.warning("Research prediction only. This probability is not a diagnosis and should not be used alone for treatment decisions.")
        st.subheader("Predictor set used")
        st.write(", ".join(features))

# ----------------------------
# Model Performance
# ----------------------------
if page == "📊 Model Performance":
    st.markdown("<div class='main-title'>Model Performance</div><div class='subtitle'>Comparison of Logistic Regression, Random Forest and ANN on a held-out test set.</div>", unsafe_allow_html=True)
    target_options=[("In-hospital mortality",mortality_col),("28-day mortality",mortality28_col),("6-month mortality",mortality6_col)]
    available=[x for x in target_options if x[1]]
    if not available:
        st.warning("No mortality outcome was detected.")
    else:
        target_label=st.selectbox("Evaluation outcome",[x[0] for x in available]); target=dict(available)[target_label]
        result=run_models(target)
        if result is None: st.warning("Model evaluation could not be completed because the selected outcome does not have enough usable data.")
        else:
            metrics,artifacts,(Xte,yte),features=result
            st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),use_container_width=True,hide_index=True)
            c1,c2=st.columns(2)
            with c1:
                st.subheader("ROC curves")
                fig=go.Figure()
                for name,(model,pred,prob) in artifacts.items():
                    fpr,tpr,_=roc_curve(yte,prob); auc=roc_auc_score(yte,prob); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{name} AUC={auc:.3f}"))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random")); fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",height=430); show_plot(fig)
            with c2:
                st.subheader("Precision–Recall curves")
                fig=go.Figure()
                for name,(model,pred,prob) in artifacts.items():
                    precision,recall,_=precision_recall_curve(yte,prob); ap=average_precision_score(yte,prob); fig.add_trace(go.Scatter(x=recall,y=precision,mode="lines",name=f"{name} PR-AUC={ap:.3f}"))
                fig.update_layout(xaxis_title="Recall",yaxis_title="Precision",height=430); show_plot(fig)
            for name,(model,pred,prob) in artifacts.items():
                with st.expander(f"{name} confusion matrix"):
                    cm=confusion_matrix(yte,pred); fig=px.imshow(cm,text_auto=True,labels={"x":"Predicted","y":"Actual","color":"Patients"},title=name); show_plot(fig,350)
            st.subheader("Model inputs")
            st.dataframe(pd.DataFrame({"Feature":features,"Used":True}),use_container_width=True,hide_index=True)

# ----------------------------
# Patient Profile is integrated into Patient Explorer; keep a separate profile page alias.
# ----------------------------
# (The sidebar uses Patient Explorer so users get filters and a full patient record.)

# ----------------------------
# Insights & Key Takeaways
# ----------------------------
if page == "⭐ Insights & Key Takeaways":
    st.markdown("<div class='main-title'>Insights & Key Takeaways</div><div class='subtitle'>Evidence summaries generated from the loaded dataset.</div>", unsafe_allow_html=True)
    insights=[]
    if gender_col:
        g=df[gender_col].value_counts(); insights.append(f"The dataset contains {g.size} detected gender categories; the largest category contains {int(g.iloc[0]):,} patients.")
    if nyha_col and mortality_col:
        t=safe_group_rate(df,nyha_col,mortality_col)
        if not t.empty: insights.append(f"In-hospital mortality varies across detected NYHA groups; the dashboard table shows the observed rate and patient count for each group.")
    if clinical.get("hs-CRP") or clinical.get("WBC") or clinical.get("NLR"):
        insights.append("The inflammation + albumin feature combines the project thresholds hs-CRP >5, WBC >10, NLR >6 and albumin <35 when those measurements are available.")
    if "CCI Group" in df.columns: insights.append("CCI is grouped into 0–2, 3–4 and 5+ for comorbidity-stratified exploration.")
    if mortality28_col: insights.append("A separate 28-day mortality outcome is available for predictive evaluation when its values contain both outcome classes.")
    for x in insights: st.markdown(f"• {x}")
    st.subheader("What the dashboard should be used for")
    st.markdown("1. Identify patterns worth clinical review.\n2. Compare patient subgroups consistently.\n3. Examine missingness before interpreting biomarkers.\n4. Compare predictive models using multiple metrics rather than one score.\n5. Treat model probabilities as research outputs, not clinical decisions.")

# ----------------------------
# Conclusion
# ----------------------------
if page == "🏁 Conclusion":
    st.markdown("<div class='main-title'>Conclusion</div><div class='subtitle'>End-to-end project summary.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'><b>Conclusion framework</b><br>
    The project brings together cleaned patient-level heart-failure data, engineered clinical features, descriptive outcome analysis, 30 prescriptive questions and predictive modeling. The dashboard makes these analyses easier to inspect interactively while keeping the distinction between observed association and prediction.</div>
    """,unsafe_allow_html=True)
    st.subheader("From data to decision support")
    st.markdown("**Clean → Engineer → Describe → Compare → Predict → Review → Communicate**")
    st.subheader("Important limitation")
    st.write("The dashboard is an analytical and research interface. Model performance depends on the cleaned dataset, outcome definition, missingness, feature availability and train/test split. Predictions should not be interpreted as clinical diagnoses or treatment recommendations.")

