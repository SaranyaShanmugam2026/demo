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
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)

st.set_page_config(page_title="HeartCare Clinical Explorer", page_icon="❤️", layout="wide")

# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
.stApp {background:#F4F9FB;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#073B4C,#0B5D6B,#087F5B);}
section[data-testid="stSidebar"] * {color:white !important;}
.hdr {background:linear-gradient(90deg,#073B4C,#087F5B);color:white;padding:25px 30px;border-radius:16px;margin-bottom:20px;}
.hdr h1 {margin:0;font-size:30px}.hdr p{margin:6px 0 0;opacity:.92}
.section {background:white;padding:20px;border-radius:15px;box-shadow:0 3px 12px rgba(0,0,0,.06);margin-bottom:18px;}
.kpi {background:white;padding:16px;border-radius:14px;border-left:5px solid #087F5B;box-shadow:0 3px 12px rgba(0,0,0,.06);min-height:105px;}
.kpi .i{font-size:25px}.kpi .t{font-size:13px;color:#637B83;font-weight:600}.kpi .v{font-size:26px;color:#073B4C;font-weight:700}
.question {background:#F8FBFC;border-left:5px solid #0B7A75;padding:15px;border-radius:10px;margin:8px 0;}
.note {background:#EAF5F8;border-left:5px solid #087F9B;padding:14px;border-radius:9px;}
.small {font-size:13px;color:#607781;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- DATA -----------------------------
DATA_FILE = Path(__file__).parent / "Cardiac_Cleaned_Data.xlsb"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE, engine="pyxlsb")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load Cardiac_Cleaned_Data.xlsb: {e}")
    st.stop()


def find_col(names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n and n.lower() in lookup:
            return lookup[n.lower()]
    return None


def first_contains(words):
    for c in df.columns:
        s = str(c).lower()
        if any(w.lower() in s for w in words):
            return c
    return None

id_col = find_col(["inpatient_number"]) or df.columns[0]
age_col = find_col(["age", "age_years", "age_in_years"])
height_col = find_col(["height", "height_m", "height_cm"])
weight_col = find_col(["weight", "weight_kg"])
agecat_col = find_col(["agecat", "age_category"])
gender_col = find_col(["gender", "sex"])
bmi_col = find_col(["bmi"])
nyha_col = find_col(["nyha_cardiac_function_classification", "nyha_cardiac", "nyha", "nyha_class"])
killip_col = find_col(["killip_grade", "killip", "killip_class"])
hf_type_col = find_col(["type_of_heart_failure", "heart_failure_type"])
cci_col = find_col(["cci", "charlson_comorbidity_index", "comorbidity_index"])
admission_col = find_col(["admission_way", "admission_type"])
destination_col = find_col(["destinationdischarge", "discharge_destination"])
los_col = find_col(["dischargeday", "length_of_stay"])
outcome_col = find_col(["outcome_during_hospitalization"])

# Outcomes: robustly normalize 0/1, Yes/No and common text.
def binary_series(col):
    if not col or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        vals = set(x.dropna().unique())
        if vals.issubset({0,1}): return x.astype(float)
    t = s.astype(str).str.strip().str.lower()
    yes = t.isin(["1","yes","y","true","dead","death","died","positive"])
    no = t.isin(["0","no","n","false","alive","negative"])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[yes] = 1
    out.loc[no] = 0
    return out

if outcome_col:
    df["_in_hospital_death"] = df[outcome_col].astype(str).str.strip().str.lower().eq("dead").astype(int)
else:
    df["_in_hospital_death"] = binary_series(find_col(["in_hospital_mortality","hospital_mortality"]))

outcome_cols = {
    "In-hospital mortality": "_in_hospital_death",
    "28-day mortality": find_col(["death_within_28_days", "mortality_28d", "mortality_28_day", "28_day_mortality"]),
    "3-month mortality": find_col(["death_within_3_months", "mortality_3_months"]),
    "6-month mortality": find_col(["death_within_6_months", "mortality_6_months"]),
    "28-day readmission": find_col(["re_admission_within_28_days", "readmission_within_28_days"]),
    "3-month readmission": find_col(["re_admission_within_3_months", "readmission_within_3_months"]),
    "6-month readmission": find_col(["re_admission_within_6_months", "readmission_within_6_months"]),
    "6-month ED return": find_col(["return_to_emergency_department_within_6_months", "ed_return_6_months"]),
}

# Derived patient-level features used throughout the dashboard.
def numeric(col):
    return pd.to_numeric(df[col], errors="coerce") if col else pd.Series(np.nan, index=df.index)

def norm_text(col):
    return df[col].astype(str).str.strip() if col else pd.Series("Unknown", index=df.index)

if bmi_col:
    b = numeric(bmi_col)
    df["BMI Category"] = pd.cut(b, [-np.inf,18.5,25,30,np.inf], labels=["Underweight","Normal","Overweight","Obese"])

if cci_col:
    c = numeric(cci_col)
    df["CCI Group"] = pd.cut(c, [-np.inf,2,4,np.inf], labels=["0–2","3–4","5+"])

# Safe string output avoids NumPy 2.x mixed-dtype np.select errors.
crp_col = find_col(["hs_crp","hs-crp","hsCRP","crp"])
wbc_col = find_col(["wbc"])
nlr_col = find_col(["nlr"])
albumin_col = find_col(["albumin"])
if crp_col or wbc_col or nlr_col:
    flags = []
    for col, threshold in [(crp_col,5),(wbc_col,10),(nlr_col,6)]:
        if col:
            flags.append(numeric(col) > threshold)
    if flags:
        df["Inflammation Flag"] = pd.concat(flags, axis=1).any(axis=1).astype(int)
    else: df["Inflammation Flag"] = 0
else: df["Inflammation Flag"] = 0

df["Low Albumin Flag"] = (numeric(albumin_col) < 35).astype(int) if albumin_col else 0
conds = [
    df["Inflammation Flag"].eq(0) & df["Low Albumin Flag"].eq(0),
    df["Inflammation Flag"].eq(1) & df["Low Albumin Flag"].eq(0),
    df["Inflammation Flag"].eq(0) & df["Low Albumin Flag"].eq(1),
    df["Inflammation Flag"].eq(1) & df["Low Albumin Flag"].eq(1),
]
df["Inflammation + Albumin Group"] = pd.Series("Unknown", index=df.index, dtype="string")
df.loc[conds[0], "Inflammation + Albumin Group"] = "Neither"
df.loc[conds[1], "Inflammation + Albumin Group"] = "Inflamed only"
df.loc[conds[2], "Inflammation + Albumin Group"] = "Low albumin only"
df.loc[conds[3], "Inflammation + Albumin Group"] = "Both"

# Common clinical columns.
biomarker_candidates = [
    "hs_crp","albumin","brain_natriuretic_peptide","creatinine_enzymatic_method","urea",
    "cystatin","egfr","d_dimer","sodium","potassium","wbc","nlr","hemoglobin","rdw",
    "lactate","oxygen_saturation","ph","troponin","high_sensitivity_troponin","ck_mb",
    "myoglobin","total_bilirubin","ast_alt_ratio","lvedd_mm","left_ventricular_end_diastolic_diameter_lv",
    "lvef","tricuspid_valve_return_pressure","ea"
]
biomarkers = [find_col([x]) for x in biomarker_candidates]
biomarkers = [x for x in biomarkers if x]

# ----------------------------- NAVIGATION -----------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center;font-size:48px'>❤️</div><h2 style='text-align:center'>HeartCare</h2><p style='text-align:center'>Clinical Explorer</p>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", [
        "🏠 Introduction","📊 Overview","🧹 Data Cleaning","🧬 Feature Engineering",
        "📈 Descriptive Analysis","💡 Prescriptive Analysis","🤖 Predictive Analytics",
        "📊 Model Performance","👤 Patient Explorer","⭐ Insights & Key Takeaways","🏁 Conclusion"
    ])

st.markdown("<div class='hdr'><h1>❤️ Heart Failure Clinical Analytics</h1><p>Descriptive analysis • Prescriptive clinical review flags • Predictive analytics • Patient exploration</p></div>", unsafe_allow_html=True)


def kpi(icon,title,value):
    st.markdown(f"<div class='kpi'><div class='i'>{icon}</div><div class='t'>{title}</div><div class='v'>{value}</div></div>", unsafe_allow_html=True)


def pct(col):
    if not col: return None
    s = binary_series(col) if col != "_in_hospital_death" else df[col]
    return s.mean()*100 if s.notna().any() else None


def fmt_pct(x): return "N/A" if x is None or pd.isna(x) else f"{x:.1f}%"


def rate_by_group(data, group_col, outcome_col):
    if not group_col or not outcome_col or group_col not in data.columns: return pd.DataFrame()
    y = binary_series(outcome_col) if outcome_col != "_in_hospital_death" else pd.to_numeric(data[outcome_col], errors="coerce")
    tmp = pd.DataFrame({"Group":data[group_col].astype(str),"Outcome":y})
    tmp = tmp.dropna(subset=["Outcome"])
    if tmp.empty: return tmp
    return tmp.groupby("Group",dropna=False)["Outcome"].agg(Patients="size", Rate="mean").reset_index().sort_values("Rate",ascending=False)

# ----------------------------- INTRODUCTION -----------------------------
if page == "🏠 Introduction":
    st.title("Introduction")
    st.markdown("""
    <div class='section'>
    <h3>Project purpose</h3>
    This dashboard presents the heart-failure dataset as an end-to-end clinical analytics project. It connects the cleaned patient-level data to descriptive analysis, question-driven prescriptive review flags, predictive modelling, model performance and patient-level exploration.
    <br><br><b>Research focus:</b> Can demographic, clinical, cardiac, laboratory, nutritional and responsiveness characteristics be combined to predict mortality outcomes among patients with heart failure?
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: st.info("**Descriptive analytics**\n\nWhat happened in this patient population?")
    with c2: st.info("**Predictive analytics**\n\nCan available baseline characteristics identify mortality risk?")
    with c3: st.info("**Prescriptive analytics**\n\nWhich observed patterns can become review or follow-up flags?")
    st.subheader("Dashboard flow")
    st.write("Raw source tables → cleaning → patient-level features → descriptive findings → prescriptive question analyses → predictive modelling → model performance → consolidated patient profile → insights → conclusion.")
    st.caption("Prescriptive outputs are analytical review flags based on observed associations; they are not treatment orders or causal conclusions.")

# ----------------------------- OVERVIEW -----------------------------
elif page == "📊 Overview":
    st.title("Overview")
    st.caption("Headline KPIs focus on cohort size, heart-failure severity, mortality, readmission and emergency return. Age, BMI, height and weight are supporting population descriptors shown below.")
    metrics = [
        ("👥","Total Patients",f"{df[id_col].nunique():,}"),
        ("☠️","In-Hospital Mortality",fmt_pct(pct("_in_hospital_death"))),
        ("🔴","28-Day Mortality",fmt_pct(pct(outcome_cols["28-day mortality"]))),
        ("🟠","6-Month Mortality",fmt_pct(pct(outcome_cols["6-month mortality"]))),
        ("🔁","28-Day Readmission",fmt_pct(pct(outcome_cols["28-day readmission"]))),
        ("🔄","6-Month Readmission",fmt_pct(pct(outcome_cols["6-month readmission"]))),
        ("🚑","6-Month ED Return",fmt_pct(pct(outcome_cols["6-month ED return"]))),
    ]
    # NYHA severe percentage
    if nyha_col:
        n = norm_text(nyha_col).str.extract(r"(\d+)",expand=False)
        severe = pd.to_numeric(n,errors="coerce").isin([3,4]).mean()*100
        metrics.insert(1,("❤️","NYHA III–IV",f"{severe:.1f}%"))
    for start in range(0,len(metrics),4):
        cols=st.columns(4)
        for j,m in enumerate(metrics[start:start+4]):
            with cols[j]: kpi(*m)
    st.subheader("Population profile")
    st.caption("Age is shown as an age group when the cleaned file contains age categories. Numeric summaries below are explicitly labelled as median values; they are not hidden calculations.")
    p1,p2,p3,p4=st.columns(4)
    if age_col:
        age_vals=numeric(age_col).dropna()
        with p1: kpi("🎂","Median Age",f"{age_vals.median():.1f} years" if len(age_vals) else "N/A")
    elif agecat_col:
        with p1: kpi("🎂","Age Variable","Age group")
    if bmi_col:
        bmi_vals=numeric(bmi_col).dropna()
        with p2: kpi("⚖️","Median BMI",f"{bmi_vals.median():.1f} kg/m²" if len(bmi_vals) else "N/A")
    if height_col:
        h=numeric(height_col).dropna()
        # Height is displayed with units so values such as 1.33 are not mistaken for an age or score.
        h_display = h.median() if len(h) else np.nan
        if len(h) and h_display > 3: h_display = h_display / 100
        with p3: kpi("📏","Median Height",f"{h_display:.2f} m" if pd.notna(h_display) else "N/A")
    if weight_col:
        w=numeric(weight_col).dropna()
        with p4: kpi("⚖️","Median Weight",f"{w.median():.1f} kg" if len(w) else "N/A")

    c1,c2 = st.columns(2)
    with c1:
        if agecat_col:
            s=norm_text(agecat_col).value_counts(dropna=False).reset_index(); s.columns=["Age group","Patients"]
            st.plotly_chart(px.bar(s,x="Age group",y="Patients",title="Age-group distribution"),use_container_width=True)
        elif age_col:
            st.plotly_chart(px.histogram(pd.DataFrame({"Age":numeric(age_col)}).dropna(),x="Age",nbins=15,title="Age distribution (years)"),use_container_width=True)
        else:
            st.info("No numeric age or age-category column was detected in the cleaned file.")
    with c2:
        if bmi_col:
            s=df["BMI Category"].astype(str).value_counts().reset_index(); s.columns=["BMI Category","Patients"]
            st.plotly_chart(px.bar(s,x="BMI Category",y="Patients",title="BMI categories"),use_container_width=True)
        else:
            st.info("BMI was not detected in the cleaned file.")

    st.markdown("<div class='note'><b>How to read the patient measurements:</b> values are shown in their source units. For example, a value such as <b>1.33</b> under Height means approximately <b>1.33 metres</b>; it is not a mean, median, or risk score. The dashboard now labels height and other measurements explicitly.</div>",unsafe_allow_html=True)

    st.subheader("Outcome snapshot")
    rows=[]
    for label,col in outcome_cols.items():
        r=pct(col)
        if r is not None: rows.append([label,r])
    if rows:
        od=pd.DataFrame(rows,columns=["Outcome","Rate"])
        st.plotly_chart(px.bar(od,x="Outcome",y="Rate",text="Rate",title="Observed outcome rates (%)"),use_container_width=True)

# ----------------------------- DATA CLEANING -----------------------------
elif page == "🧹 Data Cleaning":
    st.title("Data Cleaning")
    st.markdown("<div class='section'><b>Cleaning objective:</b> one row per patient, consistent outcome fields, usable clinical variables and transparent missingness. The dashboard reports the state of the cleaned file rather than silently changing values.</div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1:kpi("👥","Rows",f"{len(df):,}")
    with c2:kpi("🆔","Unique Patients",f"{df[id_col].nunique():,}")
    with c3:kpi("📋","Columns",f"{df.shape[1]:,}")
    with c4:kpi("🕳️","Missing Cells",f"{df.isna().mean().mean()*100:.1f}%")
    miss=(df.isna().mean()*100).sort_values(ascending=False).head(15).reset_index(); miss.columns=["Column","Missing %"]
    st.plotly_chart(px.bar(miss.sort_values("Missing %"),x="Missing %",y="Column",orientation="h",title="Top missingness rates"),use_container_width=True)
    st.subheader("Cleaning checks")
    audit=pd.DataFrame({
        "Check":["Duplicate patient IDs","Duplicate full rows","Patients with complete ID","Columns with >50% missing"],
        "Result":[int(df[id_col].duplicated().sum()),int(df.duplicated().sum()),int(df[id_col].notna().sum()),int((df.isna().mean()>0.50).sum())]
    })
    st.dataframe(audit,use_container_width=True,hide_index=True)
    st.caption("High missingness is shown for transparency. Missing clinical measurements should not automatically be interpreted as normal values.")

# ----------------------------- FEATURE ENGINEERING -----------------------------
elif page == "🧬 Feature Engineering":
    st.title("Feature Engineering")
    st.markdown("<div class='section'>Features shown here are derived to support clinical stratification and analysis. They are descriptive transformations, not new clinical diagnoses.</div>",unsafe_allow_html=True)
    derived=[c for c in ["BMI Category","CCI Group","Inflammation Flag","Low Albumin Flag","Inflammation + Albumin Group"] if c in df.columns]
    if derived: st.dataframe(df[derived].head(15),use_container_width=True)
    if "Inflammation + Albumin Group" in df:
        g=df["Inflammation + Albumin Group"].value_counts().reset_index();g.columns=["Group","Patients"]
        st.plotly_chart(px.bar(g,x="Group",y="Patients",title="Inflammation + albumin groups"),use_container_width=True)
    st.markdown("**Examples represented:** BMI category, CCI group and combined inflammation/albumin status. Additional engineered variables from the notebook can be incorporated when their exact source columns are available in the cleaned file.")

# ----------------------------- DESCRIPTIVE -----------------------------
elif page == "📈 Descriptive Analysis":
    st.title("Descriptive Analysis")
    st.caption("Use the controls to explore how patient characteristics and biomarkers are distributed. This section describes the cohort; it does not claim causation.")
    f1,f2,f3=st.columns(3)
    with f1:
        selected_bio=st.selectbox("Biomarker / clinical measure", biomarkers if biomarkers else list(df.select_dtypes(include=np.number).columns))
    with f2:
        grouping=st.selectbox("Group by", [c for c in [gender_col,agecat_col,nyha_col,killip_col,hf_type_col,"BMI Category","CCI Group",destination_col] if c])
    with f3:
        outcome_choice=st.selectbox("Outcome",[k for k,v in outcome_cols.items() if v])
    c1,c2=st.columns(2)
    with c1:
        x=numeric(selected_bio).dropna()
        if len(x): st.plotly_chart(px.histogram(pd.DataFrame({"Value":x}),x="Value",nbins=25,title=f"Distribution: {selected_bio}"),use_container_width=True)
    with c2:
        tmp=pd.DataFrame({"Group":norm_text(grouping),"Value":numeric(selected_bio)})
        tmp=tmp.dropna(subset=["Value"])
        if not tmp.empty: st.plotly_chart(px.box(tmp,x="Group",y="Value",points=False,title=f"{selected_bio} by group"),use_container_width=True)
    out=outcome_cols[outcome_choice]
    if grouping and out:
        r=rate_by_group(df,grouping,out)
        if not r.empty: st.plotly_chart(px.bar(r,x="Group",y="Rate",text="Patients",title=f"{outcome_choice} by {grouping}"),use_container_width=True)

# ----------------------------- PRESCRIPTIVE -----------------------------
elif page == "💡 Prescriptive Analysis":
    st.title("Prescriptive Analysis")
    st.caption("The 30 research questions from the project are represented as question-driven analyses. The dashboard shows the observed analysis where the required variables exist and frames the proposed action as a review consideration, not a treatment instruction.")
    questions=[
("Q1 • NYHA and outcomes","Does a higher NYHA class lead to more readmissions and deaths at 28 days, 3 months and 6 months?",nyha_col,"28-day mortality"),
("Q2 • Killip and mortality","At which Killip grade does in-hospital and 28-day death rise?",killip_col,"28-day mortality"),
("Q3 • LV structure and filling","Are enlarged LVEDD and abnormal E/A associated with 6-month outcomes?",find_col(["lvedd_mm","left_ventricular_end_diastolic_diameter_lv"]),"6-month mortality"),
("Q4 • Heart-failure type","Do left-, right- and both-sided heart failure differ in pressure, BNP and outcomes?",hf_type_col,"6-month mortality"),
("Q5 • Prior vascular disease","Do prior MI/PVD patients show more cardiac injury and different outcomes?",find_col(["myocardial_infarction","peripheral_vascular_disease"]),"28-day mortality"),
("Q6 • BNP and outcomes","At what BNP level does readmission/death risk rise?",find_col(["brain_natriuretic_peptide"]),"6-month mortality"),
("Q7 • Cardiac injury","Do high troponin/CK-MB/myoglobin identify higher mortality risk?",find_col(["troponin","high_sensitivity_troponin","ck_mb","myoglobin"]),"28-day mortality"),
("Q8 • Inflammation + albumin","Do inflammation markers together with low albumin identify higher-risk patients?","Inflammation + Albumin Group","28-day mortality"),
("Q9 • Anemia","Are anemic patients in worse NYHA classes and at higher risk?",find_col(["hemoglobin"]),"6-month mortality"),
("Q10 • Electrolytes","Which sodium/potassium ranges are linked with death/readmission?",find_col(["sodium"]),"6-month mortality"),
("Q11 • D-dimer/INR","Is high D-dimer linked to higher death?",find_col(["d_dimer","d-dimer"]),"6-month mortality"),
("Q12 • Liver congestion","Do bilirubin/AST-ALT/albumin patterns relate to congestion and outcomes?",find_col(["total_bilirubin","bilirubin"]),"6-month mortality"),
("Q13 • Blood gas","Do high lactate, low pH and low oxygen saturation identify in-hospital death risk?",find_col(["lactate"]),"In-hospital mortality"),
("Q14 • Kidney function","Does worse kidney function increase readmission and death?",find_col(["egfr","creatinine_enzymatic_method","urea","cystatin"]),"6-month mortality"),
("Q15 • Heart + kidney burden","Which patients have both kidney problems and abnormal heart markers?",find_col(["creatinine_enzymatic_method"]),"6-month mortality"),
("Q16 • Acute kidney failure vs CKD","Is acute kidney failure more dangerous than long-standing kidney disease alone?",find_col(["acute_kidney","acute_kidney_injury","aki"]),"In-hospital mortality"),
("Q17 • CCI and outcomes","Does outcome risk rise across CCI bands?","CCI Group","6-month mortality"),
("Q18 • Comorbidity + medicines","Which patients combine high comorbidity and high medication burden?",cci_col,"6-month mortality"),
("Q19 • COPD/respiratory failure","Do COPD/type II respiratory failure patients have higher CO2, oxygen needs and readmission?",find_col(["copd","type_ii_respiratory_failure"]),"6-month readmission"),
("Q20 • Ventilation","How do ventilated patients differ in severity, biomarkers and outcomes?",find_col(["respiratory_support","ventilation","mechanical_ventilation"]),"In-hospital mortality"),
("Q21 • Neurological disease","Are neurological-condition patients more likely to have non-home discharge and death?",find_col(["dementia","stroke","paralysis","reduced_consciousness"]),"In-hospital mortality"),
("Q22 • Guideline medicines","Do patients receiving more guideline medicines have different outcomes?",find_col(["beta_blocker","ace_inhibitor","arb","spironolactone"]),"6-month readmission"),
("Q23 • Inotropes","Do inotrope users form a higher-severity group with worse outcomes?",find_col(["milrinone","dobutamine","deslanoside"]),"6-month mortality"),
("Q24 • BP / shock index","Does low BP or high shock index relate to death?",find_col(["systolic_blood_pressure","sbp","blood_pressure_systolic"]),"In-hospital mortality"),
("Q25 • BMI + albumin","Are underweight patients at higher risk and does albumin differ by BMI?","BMI Category","6-month mortality"),
("Q26 • Age + gender + severity","Which age/gender groups have higher 6-month death after severity/comorbidity stratification?",agecat_col or age_col,"6-month mortality"),
("Q27 • Emergency vs planned admission","Do emergency admissions have longer stays and worse outcomes?",admission_col,"6-month mortality"),
("Q28 • Length of stay","Are very short stays followed by earlier readmission?",los_col,"28-day readmission"),
("Q29 • Discharge destination","Which destinations have higher readmission and ED-return rates?",destination_col,"6-month readmission"),
("Q30 • Frequent returners","Who are frequent returners and what profile do they share?",find_col(["visit_times"]),"6-month ED return")]

    for title,q,group,outcome_label in questions:
        with st.expander(title):
            st.markdown(f"<div class='question'><b>Research question:</b> {q}</div>",unsafe_allow_html=True)
            outcol=outcome_cols.get(outcome_label)
            if group and group in df.columns and outcol:
                r=rate_by_group(df,group,outcol)
                if not r.empty:
                    st.plotly_chart(px.bar(r,x="Group",y="Rate",text="Patients",title=f"Observed {outcome_label} rate by {group}"),use_container_width=True)
                    top=r.iloc[0]
                    st.write(f"**Observed analysis:** {int(top['Patients'])} patients are in the highest observed-rate group ({top['Group']}), with an observed {outcome_label.lower()} rate of {top['Rate']*100:.1f}%. This is an association in the dataset, not proof of causation.")
                else: st.info("The selected variables are present but do not contain enough usable outcome observations for this analysis.")
            else:
                st.info("The exact source variable needed for this question was not detected in the cleaned file, so the dashboard does not invent a result.")
            action=q.split("?")[-1].strip()
            st.caption("Review implication: use the observed pattern to identify patients or groups for closer review; do not interpret this dashboard as a treatment order.")

# ----------------------------- PREDICTIVE -----------------------------
# ----------------------------- PREDICTIVE -----------------------------
elif page == "🤖 Predictive Analytics":
    st.title("Predictive Analytics")

    st.caption(
        "This section answers: Can the Artificial Neural Network (ANN) "
        "estimate the probability of the selected outcome?"
    )

    # ---------------------------------------------------------
    # 1. SELECT PREDICTION TARGET
    # ---------------------------------------------------------
    target_options = [k for k, v in outcome_cols.items() if v]

    target_label = st.selectbox(
        "Prediction target",
        target_options,
        index=0 if target_options else None
    )

    target = outcome_cols[target_label] if target_options else None

    # ---------------------------------------------------------
    # 2. SELECT PREDICTOR VARIABLES
    # ---------------------------------------------------------
    numeric_candidates = [
        c for c in [
            age_col,
            bmi_col,
            nyha_col,
            killip_col,
            crp_col,
            wbc_col,
            nlr_col,
            albumin_col,
            find_col(["brain_natriuretic_peptide"]),
            find_col(["creatinine_enzymatic_method"]),
            find_col(["egfr"]),
            find_col(["hemoglobin"]),
            find_col(["lvef"]),
            find_col(["lvedd_mm"])
        ]
        if c
    ]

    numeric_candidates = list(dict.fromkeys(numeric_candidates))

    # ---------------------------------------------------------
    # 3. BUILD ANN MODEL
    # ---------------------------------------------------------
    if target and len(numeric_candidates) >= 2:

        md = df[numeric_candidates].copy()

        X = md.apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = binary_series(target)

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid].astype(int)

        if y.nunique() == 2 and y.value_counts().min() >= 5:

            # -------------------------------------------------
            # TRAIN / TEST SPLIT
            # -------------------------------------------------
            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # -------------------------------------------------
            # ANN MODEL
            # -------------------------------------------------
            model = Pipeline([
                (
                    "impute",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "ann",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        random_state=42,
                        early_stopping=True
                    )
                )
            ])

            model.fit(Xtr, ytr)

            # ANN probability for positive outcome
            proba = model.predict_proba(Xte)[:, 1]

            # -------------------------------------------------
            # 4. DASHBOARD KPIs
            # -------------------------------------------------

            # Descriptive dashboard band.
            # This is NOT a clinical treatment threshold.
            risk_20 = int((proba >= 0.20).sum())

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                kpi(
                    "👥",
                    "Test patients",
                    f"{len(yte):,}"
                )

            with c2:
                kpi(
                    "📈",
                    "Mean predicted risk",
                    f"{proba.mean() * 100:.1f}%"
                )

            with c3:
                kpi(
                    "🔴",
                    "Highest predicted risk",
                    f"{proba.max() * 100:.1f}%"
                )

            with c4:
                kpi(
                    "⚠️",
                    "Patients ≥20% risk",
                    f"{risk_20:,}"
                )

            # -------------------------------------------------
            # 5. EXPLAIN THE ANN OUTPUT
            # -------------------------------------------------

            st.markdown(
                """
                <div class='note'>
                <b>How to interpret these results</b><br><br>

                The ANN uses demographic, cardiac, laboratory and
                nutritional characteristics to estimate the probability
                of the selected outcome for each patient.<br><br>

                <b>Mean predicted risk</b> = average probability estimated
                across the test patients.<br>

                <b>Highest predicted risk</b> = highest probability
                estimated for one test patient.<br>

                <b>Patients ≥20% risk</b> = a descriptive dashboard group
                used to explore the model's risk distribution. It is not
                a validated clinical threshold.<br><br>

                These probabilities are model estimates and are not
                clinical diagnoses or treatment recommendations.
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # 6. RISK DISTRIBUTION
            # -------------------------------------------------

            st.subheader("ANN predicted-risk distribution")

            risk_df = pd.DataFrame({
                "Predicted risk (%)": proba * 100
            })

            fig = px.histogram(
                risk_df,
                x="Predicted risk (%)",
                nbins=20,
                title=f"ANN-estimated risk distribution — {target_label}",
                labels={
                    "Predicted risk (%)": "Estimated probability (%)"
                }
            )

            fig.add_vline(
                x=20,
                line_dash="dash",
                annotation_text="20% descriptive band"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -------------------------------------------------
            # 7. MODEL FLOW
            # -------------------------------------------------

            st.subheader("How the prediction works")

            st.markdown(
                """
                <div class='section'>
                <b>Patient characteristics</b>
                → Age, BMI, NYHA, Killip, BNP, creatinine,
                eGFR, albumin, inflammation markers, LVEF, LVEDD
                <br><br>
                ↓
                <br><br>
                <b>Artificial Neural Network</b>
                <br><br>
                ↓
                <br><br>
                <b>Estimated probability of the selected outcome</b>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # 8. PATIENT RISK EXPLORER
            # -------------------------------------------------

            st.subheader("Patient Risk Explorer")

            st.write(
                "Enter patient characteristics to see the probability "
                "estimated by the ANN."
            )

            vals = {}

            input_cols = st.columns(3)

            for i, c in enumerate(numeric_candidates):

                s = X[c].dropna()

                if len(s):

                    with input_cols[i % 3]:

                        vals[c] = st.number_input(
                            str(c)
                            .replace("_", " ")
                            .title(),
                            min_value=float(s.min()),
                            max_value=float(s.max()),
                            value=float(s.median())
                        )

            if st.button(
                "Calculate patient risk",
                type="primary"
            ):

                row = pd.DataFrame(
                    [vals],
                    columns=numeric_candidates
                )

                patient_probability = float(
                    model.predict_proba(row)[0, 1]
                )

                st.metric(
                    "Predicted probability",
                    f"{patient_probability * 100:.1f}%"
                )

                st.progress(
                    min(max(patient_probability, 0.0), 1.0)
                )

                st.info(
                    f"For the selected outcome ({target_label}), "
                    f"the ANN estimates a probability of "
                    f"{patient_probability * 100:.1f}% "
                    f"for these entered characteristics."
                )

                st.warning(
                    "Research model only. This probability is not a "
                    "clinical diagnosis or treatment recommendation."
                )

        else:

            st.warning(
                "The selected outcome does not have enough usable "
                "positive and negative observations for a stable "
                "demonstration model."
            )

    else:

        st.warning(
            "Not enough predictor variables were detected in the "
            "cleaned dataset to build the ANN demonstration."
        )


# ----------------------------- MODEL PERFORMANCE -----------------------------
# ----------------------------- MODEL PERFORMANCE -----------------------------
elif page == "📊 Model Performance":
    st.title("Model Performance & Model Comparison")
    st.caption("This page answers: how well do different models separate the outcome classes? Metrics are evaluated on a held-out test set and should be interpreted with the outcome prevalence in mind.")
    target_options=[k for k,v in outcome_cols.items() if v]
    target_label=st.selectbox("Evaluation target",target_options,index=0 if target_options else None)
    target=outcome_cols[target_label] if target_options else None
    feats=[c for c in [age_col,bmi_col,nyha_col,killip_col,crp_col,wbc_col,nlr_col,albumin_col,find_col(["brain_natriuretic_peptide"]),find_col(["creatinine_enzymatic_method"]),find_col(["egfr"]),find_col(["hemoglobin"]),find_col(["lvef"]),find_col(["lvedd_mm"])] if c]
    feats=list(dict.fromkeys(feats))
    if target and len(feats)>=2:
        X=df[feats].apply(pd.to_numeric,errors="coerce"); y=binary_series(target); valid=y.notna(); X=X.loc[valid]; y=y.loc[valid].astype(int)
        if y.nunique()==2 and y.value_counts().min()>=5:
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
            models={
                "Logistic Regression":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced"))]),
                "Random Forest":Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced",min_samples_leaf=3))]),
                "ANN":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,random_state=42,early_stopping=True))])}
            results=[]; fitted={}
            for name,m in models.items():
                m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int); fitted[name]=(m,p,pred)
                results.append([name,accuracy_score(yte,pred),precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0),roc_auc_score(yte,p),average_precision_score(yte,p)])
            res=pd.DataFrame(results,columns=["Model","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"])
            st.dataframe(res.style.format({c:"{:.3f}" for c in res.columns[1:]}),use_container_width=True,hide_index=True)
            st.subheader("Model comparison")
            st.plotly_chart(px.bar(res.melt(id_vars="Model",value_vars=["ROC-AUC","PR-AUC"],var_name="Metric",value_name="Score"),x="Model",y="Score",color="Metric",barmode="group",title="ROC-AUC and PR-AUC comparison"),use_container_width=True)
            chosen=st.selectbox("Model for detailed inspection",list(models.keys()))
            m,p,pred=fitted[chosen]; cm=confusion_matrix(yte,pred)
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(px.imshow(cm,text_auto=True,title=f"Confusion matrix — {chosen}",labels=dict(x="Predicted",y="Actual")),use_container_width=True)
            with c2:
                fpr,tpr,_=roc_curve(yte,p); fig=go.Figure();fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=chosen));fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance"));fig.update_layout(title=f"ROC curve — {chosen}",xaxis_title="False positive rate",yaxis_title="True positive rate");st.plotly_chart(fig,use_container_width=True)
            st.subheader("Model preference guide")
            st.markdown("""
            <div class='note'><b>How to compare the models:</b><br>
            • <b>ROC-AUC</b> measures overall ranking/separation across thresholds.<br>
            • <b>PR-AUC</b> is especially useful when the positive mortality class is uncommon.<br>
            • <b>Recall</b> shows how many observed positive cases are captured by the model.<br>
            • <b>Precision</b> shows how many predicted positive cases are actually positive.<br>
            • <b>F1</b> balances precision and recall.<br>
            For this project, model selection should be documented using the outcome prevalence and the metric(s) chosen before deployment; the dashboard does not treat one metric as universally best.
            </div>
            """,unsafe_allow_html=True)
            st.info("These are research-model comparisons. They do not establish clinical effectiveness or replace external validation, calibration and prospective evaluation.")
        else: st.warning("Not enough positive and negative outcome observations for a meaningful comparison on this target.")

# ----------------------------- PATIENT EXPLORER -----------------------------
elif page == "👤 Patient Explorer":
    st.title("Patient Profile & Clinical Explorer")
    st.caption("A consolidated patient profile brings demographics, heart-failure severity, biomarkers, comorbidity and outcomes into one interactive view. Nothing here is a diagnosis.")
    work=df.copy()

    # Filters
    f1,f2,f3,f4=st.columns(4)
    with f1:
        if gender_col:
            vals=sorted(work[gender_col].dropna().astype(str).unique()); sel=st.multiselect("Gender",vals); work=work[work[gender_col].astype(str).isin(sel)] if sel else work
    with f2:
        if agecat_col:
            vals=sorted(work[agecat_col].dropna().astype(str).unique()); sel=st.multiselect("Age group",vals); work=work[work[agecat_col].astype(str).isin(sel)] if sel else work
        elif age_col:
            amin,amax=float(pd.to_numeric(work[age_col],errors="coerce").min()),float(pd.to_numeric(work[age_col],errors="coerce").max())
            if pd.notna(amin) and pd.notna(amax):
                ar=st.slider("Age (years)",int(amin),int(amax),(int(amin),int(amax))); work=work[pd.to_numeric(work[age_col],errors="coerce").between(ar[0],ar[1])]
    with f3:
        if nyha_col:
            vals=sorted(work[nyha_col].dropna().astype(str).unique()); sel=st.multiselect("NYHA",vals); work=work[work[nyha_col].astype(str).isin(sel)] if sel else work
    with f4:
        if killip_col:
            vals=sorted(work[killip_col].dropna().astype(str).unique()); sel=st.multiselect("Killip",vals); work=work[work[killip_col].astype(str).isin(sel)] if sel else work

    f5,f6,f7,f8=st.columns(4)
    with f5:
        if "BMI Category" in work:
            vals=sorted(work["BMI Category"].dropna().astype(str).unique());sel=st.multiselect("BMI category",vals);work=work[work["BMI Category"].astype(str).isin(sel)] if sel else work
    with f6:
        if "CCI Group" in work:
            vals=sorted(work["CCI Group"].dropna().astype(str).unique());sel=st.multiselect("CCI group",vals);work=work[work["CCI Group"].astype(str).isin(sel)] if sel else work
    with f7:
        if admission_col:
            vals=sorted(work[admission_col].dropna().astype(str).unique());sel=st.multiselect("Admission type",vals);work=work[work[admission_col].astype(str).isin(sel)] if sel else work
    with f8:
        if hf_type_col:
            vals=sorted(work[hf_type_col].dropna().astype(str).unique());sel=st.multiselect("Heart-failure type",vals);work=work[work[hf_type_col].astype(str).isin(sel)] if sel else work

    st.divider()
    st.subheader("Selected cohort profile")
    k1,k2,k3,k4=st.columns(4)
    with k1: kpi("👥","Patients matching filters",f"{len(work):,}")
    if age_col:
        av=numeric(age_col).dropna(); val=f"{av.median():.1f} years" if len(av) else "N/A"
    else: val="Age group only"
    with k2: kpi("🎂","Median age",val)
    bv=numeric(bmi_col).dropna() if bmi_col else pd.Series(dtype=float)
    with k3: kpi("⚖️","Median BMI",f"{bv.median():.1f} kg/m²" if len(bv) else "N/A")
    lv=numeric(find_col(["lvef"])).dropna() if find_col(["lvef"]) else pd.Series(dtype=float)
    with k4: kpi("❤️","Median LVEF",f"{lv.median():.1f}%" if len(lv) else "N/A")

    st.subheader("Patient-level records")
    display_cols=[c for c in [id_col,gender_col,age_col,agecat_col,weight_col,height_col,bmi_col,"BMI Category",nyha_col,killip_col,hf_type_col,cci_col,"CCI Group",albumin_col,crp_col,find_col(["brain_natriuretic_peptide"]),find_col(["creatinine_enzymatic_method"]),find_col(["egfr"]),find_col(["lvef"])] if c]
    # Add outcome columns to the profile so users can connect characteristics to observed outcomes.
    display_cols += [c for c in outcome_cols.values() if c and c in work.columns and c not in display_cols]
    display_cols=list(dict.fromkeys(display_cols))
    st.dataframe(work[display_cols].head(200),use_container_width=True,hide_index=True)
    st.caption("Showing up to 200 matching records. Age, height, weight and BMI are raw patient-level measurements when available; the KPI cards above explicitly use medians.")

    st.subheader("Biomarker profile")
    if biomarkers:
        selected=st.selectbox("Select a biomarker / clinical measure",biomarkers,key="profile_biomarker")
        vals=numeric(selected).dropna()
        if len(vals):
            a,b=st.columns(2)
            with a:
                kpi("🧪","Median value",f"{vals.median():.2f}")
            with b:
                kpi("📊","Available observations",f"{len(vals):,}")
            fig=px.histogram(pd.DataFrame({"Value":vals}),x="Value",nbins=25,title=f"Distribution of {selected}")
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("No numeric observations are available for the selected measure.")

# ----------------------------- INSIGHTS -----------------------------
elif page == "⭐ Insights & Key Takeaways":
    st.title("Insights & Key Takeaways")
    st.markdown("<div class='section'><h3>What the dashboard is designed to communicate</h3><ul><li>Mortality and readmission are the primary outcome lenses.</li><li>NYHA and Killip provide clinical severity context.</li><li>Inflammation, albumin and renal markers provide complementary laboratory context.</li><li>Length of stay, discharge destination and ED return describe the hospital-transition pathway.</li><li>Predictive models estimate risk; descriptive and prescriptive analyses explain the observed patterns around that risk.</li></ul></div>",unsafe_allow_html=True)
    if nyha_col:
        r=rate_by_group(df,nyha_col,outcome_cols.get("6-month mortality"))
        if not r.empty: st.write(f"**Observed severity pattern:** the highest observed 6-month mortality rate in the NYHA groups shown is {r.iloc[0]['Rate']*100:.1f}% in {r.iloc[0]['Group']}. This is descriptive and unadjusted.")
    if "Inflammation + Albumin Group" in df and outcome_cols.get("6-month mortality"):
        r=rate_by_group(df,"Inflammation + Albumin Group",outcome_cols["6-month mortality"])
        if not r.empty: st.write(f"**Observed biomarker pattern:** the highest observed 6-month mortality rate among the inflammation/albumin groups is {r.iloc[0]['Rate']*100:.1f}% in {r.iloc[0]['Group']}.")
    st.caption("These statements describe this dataset and should not be generalized beyond the study population without further validation.")

# ----------------------------- CONCLUSION -----------------------------
elif page == "🏁 Conclusion":
    st.title("Conclusion")
    st.markdown("""
    <div class='section'>
    <h3>Project conclusion</h3>
    The completed workflow moves from a cleaned patient-level heart-failure dataset to clinical description, question-driven prescriptive review, predictive risk modelling and patient exploration. The central value of the dashboard is the consolidation of these layers: hospital users can see the cohort and outcomes first, examine the clinical factors associated with those outcomes, and then inspect how predictive models perform.
    <br><br>
    The predictive component should be treated as an experimental research model until it is externally validated, calibrated and evaluated on an independent clinical population. Observed associations in the descriptive and prescriptive sections should likewise be interpreted as signals for review rather than proof that an intervention will change an outcome.
    </div>
    """,unsafe_allow_html=True)
    st.subheader("Final dashboard structure")
    st.write("Introduction → Overview/KPIs → Data Cleaning → Feature Engineering → Descriptive Analysis → Prescriptive Analysis Q1–Q30 → Predictive Analytics → Model Performance & Model Preference → Consolidated Patient Profile → Insights & Key Takeaways → Conclusion")
