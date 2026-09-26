# ============================================================
# HEART FAILURE CLINICAL ANALYTICS & PREDICTION DASHBOARD
# ============================================================
# Descriptive + Predictive + Prescriptive/Decision-Support Analytics
# Research / analytical use only — not a clinical diagnosis tool.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HeartCare AI | Heart Failure Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f9fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e5b 0%, #087f70 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-header {
        background: linear-gradient(90deg, #075985, #0f766e);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.10);
    }

    .main-header h1 {
        margin: 0;
        font-size: 34px;
    }

    .main-header p {
        margin: 8px 0 0 0;
        font-size: 16px;
    }

    .section-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border-left: 6px solid #0f766e;
        margin-bottom: 18px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.06);
    }

    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 16px;
        border-left: 5px solid #0f766e;
        text-align: center;
        box-shadow: 0 3px 14px rgba(0,0,0,0.06);
    }

    .metric-title {
        font-size: 14px;
        color: #475569;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
        color: #0f3d4c;
    }

    .research-note {
        background: #e8f4f8;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #0284c7;
    }

    .risk-note {
        background: #fff7ed;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #f97316;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>❤️ HeartCare AI</h1>
        <p>Heart Failure Clinical Analytics, Mortality Prediction & Decision Support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================

DATA_FILE = "Cardiac_Cleaned_Data.xlsb"


@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE, engine="pyxlsb")


try:
    df = load_data()
except Exception as e:
    st.error(
        f"Could not load {DATA_FILE}. "
        f"Make sure the actual .xlsb file is in the same GitHub folder "
        f"as DashboardHeartfailure.py. Error: {e}"
    )
    st.stop()

# ============================================================
# COLUMN HELPERS
# ============================================================

def normalize_name(x):
    return (
        str(x)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
    )


NORMALIZED_COLUMNS = {normalize_name(c): c for c in df.columns}


def find_col(candidates):
    """
    Finds a column using exact normalized names first,
    then partial normalized-name matching.
    """
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in NORMALIZED_COLUMNS:
            return NORMALIZED_COLUMNS[key]

    for candidate in candidates:
        key = normalize_name(candidate)
        for normalized, original in NORMALIZED_COLUMNS.items():
            if key and (key in normalized or normalized in key):
                return original

    return None


# ============================================================
# COLUMN DETECTION
# ============================================================

patient_col = find_col([
    "inpatient_number",
    "patient_id",
    "patientid",
    "inpatient",
])

gender_col = find_col([
    "gender",
    "sex",
])

age_col = find_col([
    "age",
    "age_years",
    "agecat",
    "age_category",
])

weight_col = find_col([
    "weight",
    "weight_kg",
])

height_col = find_col([
    "height",
    "height_cm",
])

bmi_col = find_col([
    "bmi",
    "body_mass_index",
])

occupation_col = find_col([
    "occupation",
])

nyha_col = find_col([
    "nyha_class",
    "nyha",
    "nyha_grade",
])

killip_col = find_col([
    "killip_grade",
    "killip",
    "killip_class",
])

crp_col = find_col([
    "hs_crp",
    "hs-crp",
    "hscrp",
    "crp",
])

wbc_col = find_col([
    "wbc",
    "white_blood_cell",
    "white_blood_cells",
])

nlr_col = find_col([
    "nlr",
    "neutrophil_lymphocyte_ratio",
])

albumin_col = find_col([
    "albumin",
    "serum_albumin",
])

responsiveness_col = find_col([
    "responsiveness",
    "response",
    "responsive",
])

in_hospital_col = find_col([
    "in_hospital_mortality",
    "in_hospital_mortality",
    "hospital_mortality",
    "inpatient_mortality",
    "mortality_in_hospital",
    "in_hospital_death",
])

mortality_28d_col = find_col([
    "mortality_28d",
    "28_day_mortality",
    "28_days_mortality",
    "mortality_28_d",
    "28d_mortality",
    "mortality_28days",
    "28_day_death",
])

# Fallback detection for messy names
if in_hospital_col is None:
    for c in df.columns:
        n = normalize_name(c)
        if "hospital" in n and "mortality" in n:
            in_hospital_col = c
            break

if mortality_28d_col is None:
    for c in df.columns:
        n = normalize_name(c)
        if ("28" in n or "28day" in n or "28d" in n) and (
            "mortality" in n or "death" in n
        ):
            mortality_28d_col = c
            break


# ============================================================
# DATA PREPARATION
# ============================================================

def numeric_series(col):
    if col is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(df[col], errors="coerce")


def to_binary(series):
    """
    Converts common mortality/response formats to 0/1.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # Numeric 0/1
    result = numeric.copy()

    text = series.astype(str).str.strip().str.lower()

    positive = {
        "1",
        "yes",
        "y",
        "true",
        "dead",
        "death",
        "died",
        "mortality",
        "positive",
        "event",
    }

    negative = {
        "0",
        "no",
        "n",
        "false",
        "alive",
        "survived",
        "survival",
        "negative",
        "nonevent",
        "non-event",
    }

    result[text.isin(positive)] = 1
    result[text.isin(negative)] = 0

    return result


# Create clean mortality variables without modifying the original columns
if in_hospital_col:
    df["_in_hospital_target"] = to_binary(df[in_hospital_col])
else:
    df["_in_hospital_target"] = np.nan

if mortality_28d_col:
    df["_28d_target"] = to_binary(df[mortality_28d_col])
else:
    df["_28d_target"] = np.nan


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px;">
        <div style="font-size:48px;">❤️</div>
        <h2>HeartCare AI</h2>
        <p>Clinical Analytics Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    "🏥 Overview",
    "🧹 Data Quality",
    "👤 Patient Profile",
    "❤️ Cardiac Risk",
    "🧪 Biomarkers",
    "⚠️ Mortality Analysis",
    "🤖 AI Prediction",
    "📈 Model Performance",
    "💡 Decision Support",
]

page = st.sidebar.radio("NAVIGATION", pages)

st.sidebar.markdown("---")

st.sidebar.caption(
    "For research and analytical use. "
    "Predictions should not replace clinician judgment."
)

# ============================================================
# FILTERS
# ============================================================

st.sidebar.markdown("### 🔎 Filters")

filtered_df = df.copy()

if gender_col:
    values = (
        filtered_df[gender_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        values,
        default=values,
    )

    if selected_gender:
        filtered_df = filtered_df[
            filtered_df[gender_col].astype(str).isin(selected_gender)
        ]

if nyha_col:
    nyha_values = pd.to_numeric(
        filtered_df[nyha_col], errors="coerce"
    ).dropna()

    if not nyha_values.empty:
        min_nyha = int(nyha_values.min())
        max_nyha = int(nyha_values.max())

        if min_nyha < max_nyha:
            selected_nyha = st.sidebar.slider(
                "NYHA range",
                min_nyha,
                max_nyha,
                (min_nyha, max_nyha),
            )

            nyha_numeric = pd.to_numeric(
                filtered_df[nyha_col],
                errors="coerce",
            )

            filtered_df = filtered_df[
                nyha_numeric.between(
                    selected_nyha[0],
                    selected_nyha[1],
                    inclusive="both",
                )
            ]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(icon, title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:26px;">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mortality_rate(data, target_col):
    if target_col not in data.columns:
        return np.nan

    s = pd.to_numeric(data[target_col], errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return s.mean() * 100


def show_group_mortality(data, group_col, target_col, title):
    if group_col is None:
        st.info(f"{title}: required column was not detected.")
        return

    if target_col not in data.columns:
        st.info(f"{title}: mortality outcome was not detected.")
        return

    temp = data[[group_col, target_col]].copy()
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        st.info(f"No usable data available for {title}.")
        return

    summary = (
        temp.groupby(group_col)[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )

    summary["Mortality %"] = summary["mean"] * 100

    fig = px.bar(
        summary,
        x=group_col,
        y="Mortality %",
        text="Mortality %",
        title=title,
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    st.plotly_chart(fig, use_container_width=True)

    display = summary[[group_col, "count", "Mortality %"]].copy()
    display.columns = [str(x) for x in display.columns]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏥 Overview":

    st.header("🏥 Hospital Overview")
    st.subheader("Descriptive clinical analytics")

    total_patients = (
        filtered_df[patient_col].nunique()
        if patient_col
        else len(filtered_df)
    )

    ih_rate = mortality_rate(filtered_df, "_in_hospital_target")
    d28_rate = mortality_rate(filtered_df, "_28d_target")

    avg_age = (
        numeric_series(age_col).mean()
        if age_col
        else np.nan
    )

    avg_bmi = (
        numeric_series(bmi_col).mean()
        if bmi_col
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("👥", "Total Patients", f"{total_patients:,}")

    with c2:
        metric_card(
            "🏥",
            "In-Hospital Mortality",
            f"{ih_rate:.1f}%" if pd.notna(ih_rate) else "N/A",
        )

    with c3:
        metric_card(
            "📅",
            "28-Day Mortality",
            f"{d28_rate:.1f}%" if pd.notna(d28_rate) else "N/A",
        )

    with c4:
        metric_card(
            "🎂",
            "Average Age",
            f"{avg_age:.1f}" if pd.notna(avg_age) else "N/A",
        )

    st.markdown("###")

    st.markdown(
        """
        <div class="research-note">
        <b>Clinical analytics objective</b><br><br>
        This dashboard evaluates demographic, cardiac, inflammatory,
        nutritional and responsiveness characteristics in relation to
        in-hospital and 28-day mortality outcomes. It also provides an
        experimental machine-learning prediction interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Patient Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        if gender_col:
            gender_counts = (
                filtered_df[gender_col]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            gender_counts.columns = ["Gender", "Patients"]

            fig = px.pie(
                gender_counts,
                names="Gender",
                values="Patients",
                hole=0.45,
                title="Patient Distribution by Gender",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gender column not detected.")

    with col2:
        if age_col:
            age_data = numeric_series(age_col).dropna()

            if not age_data.empty:
                fig = px.histogram(
                    age_data,
                    x=age_data,
                    nbins=20,
                    title="Age Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.info("Age data unavailable.")
        else:
            st.info("Age column not detected.")


# ============================================================
# DATA QUALITY
# ============================================================

if page == "🧹 Data Quality":

    st.header("🧹 Data Quality & Dataset Review")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("📋", "Rows", f"{len(df):,}")

    with c2:
        metric_card("🧬", "Columns", f"{df.shape[1]:,}")

    with c3:
        metric_card(
            "♻️",
            "Duplicate Rows",
            f"{df.duplicated().sum():,}",
        )

    st.markdown("### Missing Values")

    missing = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing.columns = ["Column", "Missing"]

    missing["Missing %"] = (
        missing["Missing"] / len(df) * 100
    ).round(2)

    missing = missing.sort_values(
        "Missing %",
        ascending=False,
    )

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Detected Clinical Variables")

    detected = {
        "Patient ID": patient_col,
        "Gender": gender_col,
        "Age": age_col,
        "Weight": weight_col,
        "Height": height_col,
        "BMI": bmi_col,
        "Occupation": occupation_col,
        "NYHA": nyha_col,
        "Killip": killip_col,
        "hs-CRP": crp_col,
        "WBC": wbc_col,
        "NLR": nlr_col,
        "Albumin": albumin_col,
        "Responsiveness": responsiveness_col,
        "In-Hospital Mortality": in_hospital_col,
        "28-Day Mortality": mortality_28d_col,
    }

    detected_df = pd.DataFrame(
        list(detected.items()),
        columns=["Variable", "Detected Column"],
    )

    st.dataframe(
        detected_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PATIENT PROFILE
# ============================================================

if page == "👤 Patient Profile":

    st.header("👤 Patient Profile")

    if patient_col is None:
        st.warning("Patient identifier column was not detected.")
    else:

        patients = (
            filtered_df[patient_col]
            .dropna()
            .unique()
            .tolist()
        )

        if len(patients) == 0:
            st.info("No patients available after filtering.")
        else:

            selected_patient = st.selectbox(
                "Select Patient",
                patients,
            )

            patient_data = filtered_df[
                filtered_df[patient_col] == selected_patient
            ].copy()

            st.markdown("### Patient Summary")

            cols = st.columns(4)

            patient_age = (
                pd.to_numeric(
                    patient_data[age_col],
                    errors="coerce",
                ).iloc[0]
                if age_col and age_col in patient_data
                else np.nan
            )

            patient_bmi = (
                pd.to_numeric(
                    patient_data[bmi_col],
                    errors="coerce",
                ).iloc[0]
                if bmi_col and bmi_col in patient_data
                else np.nan
            )

            patient_nyha = (
                patient_data[nyha_col].iloc[0]
                if nyha_col
                else "N/A"
            )

            patient_killip = (
                patient_data[killip_col].iloc[0]
                if killip_col
                else "N/A"
            )

            with cols[0]:
                metric_card(
                    "👤",
                    "Patient ID",
                    str(selected_patient),
                )

            with cols[1]:
                metric_card(
                    "🎂",
                    "Age",
                    f"{patient_age:.1f}"
                    if pd.notna(patient_age)
                    else "N/A",
                )

            with cols[2]:
                metric_card(
                    "⚖️",
                    "BMI",
                    f"{patient_bmi:.1f}"
                    if pd.notna(patient_bmi)
                    else "N/A",
                )

            with cols[3]:
                metric_card(
                    "❤️",
                    "NYHA",
                    str(patient_nyha),
                )

            st.markdown("### Clinical Characteristics")

            profile = {}

            for label, col in [
                ("Gender", gender_col),
                ("Weight", weight_col),
                ("Height", height_col),
                ("BMI", bmi_col),
                ("Occupation", occupation_col),
                ("NYHA", nyha_col),
                ("Killip", killip_col),
                ("hs-CRP", crp_col),
                ("WBC", wbc_col),
                ("NLR", nlr_col),
                ("Albumin", albumin_col),
                ("Responsiveness", responsiveness_col),
            ]:
                if col:
                    profile[label] = patient_data[col].iloc[0]

            profile_df = pd.DataFrame(
                list(profile.items()),
                columns=["Variable", "Value"],
            )

            st.dataframe(
                profile_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# CARDIAC RISK
# ============================================================

if page == "❤️ Cardiac Risk":

    st.header("❤️ Cardiac Severity & Risk")

    st.markdown(
        """
        This section examines whether established cardiac severity
        indicators are associated with mortality outcomes.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        show_group_mortality(
            filtered_df,
            nyha_col,
            "_in_hospital_target",
            "In-Hospital Mortality by NYHA",
        )

    with col2:
        show_group_mortality(
            filtered_df,
            killip_col,
            "_in_hospital_target",
            "In-Hospital Mortality by Killip Grade",
        )

    st.markdown("### 28-Day Mortality")

    col1, col2 = st.columns(2)

    with col1:
        show_group_mortality(
            filtered_df,
            nyha_col,
            "_28d_target",
            "28-Day Mortality by NYHA",
        )

    with col2:
        show_group_mortality(
            filtered_df,
            killip_col,
            "_28d_target",
            "28-Day Mortality by Killip Grade",
        )


# ============================================================
# BIOMARKERS
# ============================================================

if page == "🧪 Biomarkers":

    st.header("🧪 Inflammatory & Nutritional Biomarkers")

    biomarker_cols = [
        ("hs-CRP", crp_col),
        ("WBC", wbc_col),
        ("NLR", nlr_col),
        ("Albumin", albumin_col),
    ]

    available_biomarkers = [
        (name, col)
        for name, col in biomarker_cols
        if col is not None
    ]

    if not available_biomarkers:
        st.warning("No biomarker columns were detected.")
    else:

        for name, col in available_biomarkers:

            st.markdown(f"### {name}")

            temp = pd.DataFrame(
                {
                    "Value": pd.to_numeric(
                        filtered_df[col],
                        errors="coerce",
                    )
                }
            ).dropna()

            if temp.empty:
                st.info(f"No usable data for {name}.")
                continue

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    temp,
                    x="Value",
                    nbins=30,
                    title=f"{name} Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            with col2:

                if "_in_hospital_target" in filtered_df:

                    temp2 = filtered_df[
                        [col, "_in_hospital_target"]
                    ].copy()

                    temp2[col] = pd.to_numeric(
                        temp2[col],
                        errors="coerce",
                    )

                    temp2["_in_hospital_target"] = pd.to_numeric(
                        temp2["_in_hospital_target"],
                        errors="coerce",
                    )

                    temp2 = temp2.dropna()

                    if len(temp2) > 0:
                        fig = px.box(
                            temp2,
                            x="_in_hospital_target",
                            y=col,
                            title=f"{name} by In-Hospital Outcome",
                            labels={
                                "_in_hospital_target": "Mortality (0=Survival, 1=Death)"
                            },
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )


    st.markdown("### 🔥 Inflammation + Albumin")

    if crp_col and albumin_col:

        combo = filtered_df[
            [crp_col, albumin_col, "_in_hospital_target"]
        ].copy()

        combo[crp_col] = pd.to_numeric(
            combo[crp_col],
            errors="coerce",
        )

        combo[albumin_col] = pd.to_numeric(
            combo[albumin_col],
            errors="coerce",
        )

        combo["_in_hospital_target"] = pd.to_numeric(
            combo["_in_hospital_target"],
            errors="coerce",
        )

        combo = combo.dropna()

        if len(combo) > 0:

            fig = px.scatter(
                combo,
                x=crp_col,
                y=albumin_col,
                color="_in_hospital_target",
                title="hs-CRP vs Albumin by In-Hospital Outcome",
                labels={
                    "_in_hospital_target": "Mortality"
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# MORTALITY ANALYSIS
# ============================================================

if page == "⚠️ Mortality Analysis":

    st.header("⚠️ Mortality Analysis")

    st.markdown(
        """
        Descriptive analysis of factors associated with in-hospital
        and 28-day mortality.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        ih_rate = mortality_rate(
            filtered_df,
            "_in_hospital_target",
        )

        metric_card(
            "🏥",
            "In-Hospital Mortality",
            f"{ih_rate:.2f}%"
            if pd.notna(ih_rate)
            else "N/A",
        )

    with col2:

        d28_rate = mortality_rate(
            filtered_df,
            "_28d_target",
        )

        metric_card(
            "📅",
            "28-Day Mortality",
            f"{d28_rate:.2f}%"
            if pd.notna(d28_rate)
            else "N/A",
        )

    st.markdown("### Demographic Factors")

    col1, col2 = st.columns(2)

    with col1:
        show_group_mortality(
            filtered_df,
            gender_col,
            "_in_hospital_target",
            "In-Hospital Mortality by Gender",
        )

    with col2:
        show_group_mortality(
            filtered_df,
            age_col,
            "_in_hospital_target",
            "In-Hospital Mortality by Age",
        )

    st.markdown("### Clinical Factors")

    col1, col2 = st.columns(2)

    with col1:
        show_group_mortality(
            filtered_df,
            nyha_col,
            "_in_hospital_target",
            "Mortality by NYHA",
        )

    with col2:
        show_group_mortality(
            filtered_df,
            killip_col,
            "_in_hospital_target",
            "Mortality by Killip",
        )


# ============================================================
# AI PREDICTION
# ============================================================

if page == "🤖 AI Prediction":

    st.header("🤖 AI Mortality Prediction")

    st.markdown(
        """
        The model uses available demographic, cardiac, inflammatory
        and nutritional variables to estimate mortality risk.
        """
    )

    target_options = {}

    if in_hospital_col:
        target_options["In-Hospital Mortality"] = "_in_hospital_target"

    if mortality_28d_col:
        target_options["28-Day Mortality"] = "_28d_target"

    if not target_options:
        st.error(
            "No mortality outcome was detected in the dataset."
        )
        st.stop()

    selected_target_label = st.selectbox(
        "Select prediction outcome",
        list(target_options.keys()),
    )

    selected_target = target_options[selected_target_label]

    candidate_features = [
        age_col,
        gender_col,
        weight_col,
        height_col,
        bmi_col,
        nyha_col,
        killip_col,
        crp_col,
        wbc_col,
        nlr_col,
        albumin_col,
        responsiveness_col,
    ]

    features = [
        c
        for c in candidate_features
        if c is not None and c != selected_target
    ]

    if len(features) < 2:
        st.error(
            "At least two usable predictor variables are required."
        )
    else:

        model_df = filtered_df[
            features + [selected_target]
        ].copy()

        # Convert every feature to numeric.
        # Categorical variables such as gender/responsiveness
        # are factorized so that the ANN can use them.
        for col in features:

            converted = pd.to_numeric(
                model_df[col],
                errors="coerce",
            )

            if converted.notna().sum() == 0:

                codes, _ = pd.factorize(
                    model_df[col].astype(str)
                )

                model_df[col] = codes.astype(float)

            else:
                model_df[col] = converted

        model_df[selected_target] = pd.to_numeric(
            model_df[selected_target],
            errors="coerce",
        )

        model_df = model_df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        # Need both outcome classes
        if model_df[selected_target].nunique() < 2:

            st.error(
                "The selected mortality outcome does not contain "
                "both survival and mortality classes after cleaning."
            )

        elif len(model_df) < 30:

            st.warning(
                "Too few complete observations are available for a "
                "stable demonstration model."
            )

        else:

            X = model_df[features]
            y = model_df[selected_target].astype(int)

            class_counts = y.value_counts()

            if class_counts.min() < 2:

                st.error(
                    "One mortality class has fewer than two observations."
                )

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(
                    X_train
                )

                X_test_scaled = scaler.transform(
                    X_test
                )

                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42,
                )

                model.fit(
                    X_train_scaled,
                    y_train,
                )

                predictions = model.predict(
                    X_test_scaled
                )

                probabilities = model.predict_proba(
                    X_test_scaled
                )[:, 1]

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                try:
                    auc = roc_auc_score(
                        y_test,
                        probabilities,
                    )
                except Exception:
                    auc = np.nan

                st.markdown("### Model Results")

                c1, c2, c3, c4, c5 = st.columns(5)

                with c1:
                    metric_card(
                        "🎯",
                        "Accuracy",
                        f"{accuracy:.3f}",
                    )

                with c2:
                    metric_card(
                        "🔎",
                        "Precision",
                        f"{precision:.3f}",
                    )

                with c3:
                    metric_card(
                        "🚨",
                        "Recall",
                        f"{recall:.3f}",
                    )

                with c4:
                    metric_card(
                        "⚖️",
                        "F1 Score",
                        f"{f1:.3f}",
                    )

                with c5:
                    metric_card(
                        "📈",
                        "ROC-AUC",
                        f"{auc:.3f}"
                        if pd.notna(auc)
                        else "N/A",
                    )

                st.markdown("### Model Inputs")

                feature_table = pd.DataFrame(
                    {
                        "Feature": features,
                        "Used in ANN": ["Yes"] * len(features),
                    }
                )

                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    predictions,
                )

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels={
                        "x": "Predicted",
                        "y": "Actual",
                        "color": "Patients",
                    },
                    x=["Survival", "Mortality"],
                    y=["Survival", "Mortality"],
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.markdown(
                    """
                    <div class="research-note">
                    <b>Interpretation:</b>
                    Recall indicates the proportion of observed mortality
                    cases identified by the model. ROC-AUC summarizes
                    discrimination across probability thresholds.
                    These metrics should be interpreted with the sample
                    size, class balance, missingness and validation design
                    in mind.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

if page == "📈 Model Performance":

    st.header("📈 ANN Model Performance")

    st.info(
        "This page trains the same ANN workflow on the selected outcome "
        "and displays evaluation metrics, ROC curve and confusion matrix."
    )

    target_options = {}

    if in_hospital_col:
        target_options["In-Hospital Mortality"] = "_in_hospital_target"

    if mortality_28d_col:
        target_options["28-Day Mortality"] = "_28d_target"

    if not target_options:
        st.error("No mortality outcome detected.")
        st.stop()

    selected_target_label = st.selectbox(
        "Outcome",
        list(target_options.keys()),
        key="performance_target",
    )

    selected_target = target_options[selected_target_label]

    candidate_features = [
        age_col,
        gender_col,
        bmi_col,
        nyha_col,
        killip_col,
        crp_col,
        wbc_col,
        nlr_col,
        albumin_col,
        responsiveness_col,
    ]

    features = [
        c
        for c in candidate_features
        if c is not None
    ]

    if len(features) < 2:
        st.error("Not enough predictor variables were detected.")
    else:

        model_df = filtered_df[
            features + [selected_target]
        ].copy()

        for col in features:

            numeric = pd.to_numeric(
                model_df[col],
                errors="coerce",
            )

            if numeric.notna().sum() == 0:

                codes, _ = pd.factorize(
                    model_df[col].astype(str)
                )

                model_df[col] = codes.astype(float)

            else:

                model_df[col] = numeric

        model_df[selected_target] = pd.to_numeric(
            model_df[selected_target],
            errors="coerce",
        )

        model_df = model_df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if model_df[selected_target].nunique() < 2:

            st.error(
                "The outcome must contain both survival and mortality."
            )

        elif len(model_df) < 30:

            st.warning(
                "At least 30 complete observations are recommended "
                "for this demonstration."
            )

        else:

            X = model_df[features]
            y = model_df[selected_target].astype(int)

            if y.value_counts().min() < 2:

                st.error(
                    "Insufficient observations in one outcome class."
                )

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(
                    X_train
                )

                X_test_scaled = scaler.transform(
                    X_test
                )

                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42,
                )

                model.fit(
                    X_train_scaled,
                    y_train,
                )

                predictions = model.predict(
                    X_test_scaled
                )

                probabilities = model.predict_proba(
                    X_test_scaled
                )[:, 1]

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )

                auc = roc_auc_score(
                    y_test,
                    probabilities,
                )

                c1, c2, c3, c4, c5 = st.columns(5)

                with c1:
                    metric_card(
                        "🎯",
                        "Accuracy",
                        f"{accuracy:.3f}",
                    )

                with c2:
                    metric_card(
                        "🔎",
                        "Precision",
                        f"{precision:.3f}",
                    )

                with c3:
                    metric_card(
                        "🚨",
                        "Recall",
                        f"{recall:.3f}",
                    )

                with c4:
                    metric_card(
                        "⚖️",
                        "F1",
                        f"{f1:.3f}",
                    )

                with c5:
                    metric_card(
                        "📈",
                        "ROC-AUC",
                        f"{auc:.3f}",
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_test,
                        predictions,
                    )

                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels={
                            "x": "Predicted",
                            "y": "Actual",
                            "color": "Patients",
                        },
                        x=["Survival", "Mortality"],
                        y=["Survival", "Mortality"],
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                with col2:

                    st.subheader("ROC Curve")

                    fpr, tpr, _ = roc_curve(
                        y_test,
                        probabilities,
                    )

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"ANN AUC = {auc:.3f}",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random",
                        )
                    )

                    fig.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=450,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                st.subheader("Model Input Features")

                feature_table = pd.DataFrame(
                    {
                        "Feature": features,
                        "Used in ANN": ["Yes"] * len(features),
                    }
                )

                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# DECISION SUPPORT / PRESCRIPTIVE ANALYTICS
# ============================================================

if page == "💡 Decision Support":

    st.header("💡 Decision-Support Analytics")

    st.markdown(
        """
        This section translates the descriptive and predictive findings
        into structured areas for clinical review. It is intended as a
        research decision-support layer, not as an automated treatment
        recommendation.
        """
    )

    st.markdown("### 1. Cardiac Severity Review")

    if nyha_col or killip_col:

        if nyha_col:
            nyha_values = pd.to_numeric(
                filtered_df[nyha_col],
                errors="coerce",
            )

            if nyha_values.notna().any():

                high_nyha = (
                    nyha_values >= nyha_values.median()
                ).mean() * 100

                st.write(
                    f"• {high_nyha:.1f}% of filtered records are "
                    "at or above the median NYHA severity."
                )

        if killip_col:
            killip_values = pd.to_numeric(
                filtered_df[killip_col],
                errors="coerce",
            )

            if killip_values.notna().any():

                high_killip = (
                    killip_values >= killip_values.median()
                ).mean() * 100

                st.write(
                    f"• {high_killip:.1f}% of filtered records are "
                    "at or above the median Killip severity."
                )

    else:
        st.info("NYHA/Killip data not detected.")

    st.markdown("### 2. Inflammation & Nutrition Review")

    available = []

    if crp_col:
        available.append("hs-CRP")

    if wbc_col:
        available.append("WBC")

    if nlr_col:
        available.append("NLR")

    if albumin_col:
        available.append("Albumin")

    if available:

        st.write(
            "Available markers for combined review: "
            + ", ".join(available)
            + "."
        )

        st.write(
            "Consider reviewing elevated inflammatory markers together "
            "with low albumin rather than interpreting a single marker "
            "in isolation."
        )

    else:
        st.info("Inflammatory/nutritional markers were not detected.")

    st.markdown("### 3. Mortality Risk Review")

    ih_rate = mortality_rate(
        filtered_df,
        "_in_hospital_target",
    )

    d28_rate = mortality_rate(
        filtered_df,
        "_28d_target",
    )

    if pd.notna(ih_rate):
        st.write(
            f"• Observed in-hospital mortality in the selected population: "
            f"{ih_rate:.1f}%."
        )

    if pd.notna(d28_rate):
        st.write(
            f"• Observed 28-day mortality in the selected population: "
            f"{d28_rate:.1f}%."
        )

    st.markdown("### 4. Suggested Review Checklist")

    checklist = pd.DataFrame(
        {
            "Domain": [
                "Demographics",
                "Cardiac severity",
                "Inflammation",
                "Nutrition",
                "Responsiveness",
                "Mortality outcome",
            ],
            "Review area": [
                "Age, gender, BMI and baseline characteristics",
                "NYHA and Killip severity",
                "hs-CRP, WBC and NLR",
                "Albumin and nutritional status",
                "Clinical responsiveness indicator",
                "In-hospital and/or 28-day mortality",
            ],
        }
    )

    st.dataframe(
        checklist,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="risk-note">
        <b>Important:</b> These are analytical review prompts, not
        individualized treatment instructions. Any clinical action should
        be based on the complete patient record and qualified clinical
        judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "HeartCare AI | Heart Failure Clinical Analytics Dashboard | "
    "Research and analytical use only"
)
