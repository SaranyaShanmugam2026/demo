# ============================================================
# HEART FAILURE CLINICAL ANALYTICS & AI DASHBOARD
# ============================================================
# Descriptive + Predictive + Data-Driven Prescriptive Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Failure Clinical Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F9FB;
}

/* Main header */
.hospital-header {
    background: linear-gradient(
        90deg,
        #075985,
        #0F766E
    );
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.10);
}

.hospital-header h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.hospital-header p {
    font-size: 16px;
    margin: 0;
}

/* Section titles */
.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #083B4C;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #527080;
    margin-bottom: 20px;
}

/* KPI cards */
.metric-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    min-height: 145px;
    border-left: 6px solid #0F766E;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

.metric-icon {
    font-size: 30px;
    margin-bottom: 5px;
}

.metric-title {
    font-size: 14px;
    color: #557080;
    font-weight: 600;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
    color: #083B4C;
    margin-top: 8px;
}

/* Information boxes */
.info-box {
    background: #EAF5F8;
    border-left: 6px solid #087F9B;
    padding: 18px;
    border-radius: 10px;
    margin: 15px 0;
}

.insight-box {
    background: #EEF8F3;
    border-left: 6px solid #0F766E;
    padding: 18px;
    border-radius: 10px;
    margin: 12px 0;
}

.warning-box {
    background: #FFF7E6;
    border-left: 6px solid #D97706;
    padding: 18px;
    border-radius: 10px;
    margin: 12px 0;
}

.risk-box {
    background: #FEF2F2;
    border-left: 6px solid #DC2626;
    padding: 18px;
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #064E5B,
        #087F6B
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Tables */
.dataframe {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hospital-header">

<h1>❤️ Heart Failure Clinical Analytics & AI</h1>

<p>
Descriptive Analytics • Mortality Risk Analysis •
Artificial Neural Network Prediction • Data-Driven Insights
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        "Cardiac_Cleaned_Data.xlsb",
        engine="pyxlsb"
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Could not load Cardiac_Cleaned_Data.xlsb: {e}"
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]

# Remove completely empty rows
df = df.dropna(how="all").copy()


# ============================================================
# COLUMN DETECTION
# ============================================================

def normalize_column(name):

    return (
        str(name)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


normalized_columns = {
    normalize_column(c): c
    for c in df.columns
}


def find_column(possible_names):

    for name in possible_names:

        normalized_name = normalize_column(name)

        if normalized_name in normalized_columns:

            return normalized_columns[normalized_name]

    return None


# ------------------------------------------------------------
# Demographics
# ------------------------------------------------------------

patient_col = find_column([
    "inpatient_number",
    "patient_id",
    "patient_number",
    "id"
])

age_col = find_column([
    "age",
    "age_years"
])

gender_col = find_column([
    "gender",
    "sex"
])

weight_col = find_column([
    "weight",
    "body_weight"
])

height_col = find_column([
    "height",
    "body_height"
])

bmi_col = find_column([
    "bmi",
    "BMI"
])

occupation_col = find_column([
    "occupation"
])

agecat_col = find_column([
    "agecat",
    "age_category"
])


# ------------------------------------------------------------
# Cardiac
# ------------------------------------------------------------

nyha_col = find_column([
    "nyha_cardiac",
    "nyha",
    "nyha_class"
])

killip_col = find_column([
    "killip_grade",
    "killip",
    "killip_class"
])


# ------------------------------------------------------------
# Biomarkers
# ------------------------------------------------------------

crp_col = find_column([
    "hs_crp",
    "hs-crp",
    "hsCRP",
    "crp"
])

wbc_col = find_column([
    "wbc",
    "WBC"
])

nlr_col = find_column([
    "nlr",
    "NLR"
])

albumin_col = find_column([
    "albumin",
    "Albumin"
])


# ------------------------------------------------------------
# Mortality
# ------------------------------------------------------------

mortality_col = find_column([
    "in_hospital_mortality",
    "hospital_mortality",
    "in_hospital_death",
    "mortality",
    "death"
])

mortality28_col = find_column([
    "mortality_28d",
    "mortality_28_day",
    "28_day_mortality",
    "mortality_28dincrease_28d",
    "increase_28d"
])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(icon, title, value):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-icon">
                {icon}
            </div>

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def numeric_series(column):

    if column is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def binary_target(series):

    if series is None:
        return None

    s = series.copy()

    # Numeric first
    numeric = pd.to_numeric(
        s,
        errors="coerce"
    )

    if numeric.notna().sum() > 0:

        unique_values = set(
            numeric.dropna().unique()
        )

        if unique_values.issubset({0, 1}):

            return numeric.astype("Int64")

    # Text values
    text = (
        s.astype(str)
        .str.lower()
        .str.strip()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "dead": 1,
        "alive": 0,
        "death": 1,
        "survival": 0,
        "died": 1,
        "survived": 0,
        "true": 1,
        "false": 0
    }

    return text.map(mapping).astype("Int64")


def mortality_rate(column, data=None):

    if column is None:
        return np.nan

    if data is None:
        data = df

    values = binary_target(
        data[column]
    )

    if values is None:
        return np.nan

    return values.astype(float).mean() * 100


def display_rate(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.1f}%"


def safe_mean(column):

    if column is None:
        return np.nan

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).mean()


def group_mortality(data, group_col, target_col):

    if group_col is None or target_col is None:
        return pd.DataFrame()

    temp = data[
        [group_col, target_col]
    ].copy()

    temp["target_binary"] = binary_target(
        temp[target_col]
    )

    temp = temp.dropna(
        subset=[group_col, "target_binary"]
    )

    if temp.empty:
        return pd.DataFrame()

    result = (
        temp
        .groupby(group_col)["target_binary"]
        .agg(
            Patients="count",
            Mortality_Rate="mean"
        )
        .reset_index()
    )

    result["Mortality_Rate"] *= 100

    return result


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        text-align:center;
        padding:15px;
        ">

        <div style="font-size:50px;">
        ❤️
        </div>

        <h2>
        HeartCare AI
        </h2>

        <p>
        Clinical Analytics & Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏥 Overview",
            "📊 Data Quality",
            "👤 Patient Profile",
            "❤️ Cardiac Risk",
            "🧪 Biomarkers & Nutrition",
            "⚠️ Mortality Analysis",
            "🔎 Relationship Explorer",
            "🤖 AI Prediction",
            "📈 Model Performance",
            "💡 Data-Driven Insights",
            "📋 Research Questions"
        ]
    )

    st.divider()

    st.caption(
        "Heart Failure Clinical Analytics"
    )

    st.caption(
        "For research and analytical use."
    )


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "🏥 Overview":

    st.markdown(
        '<div class="main-title">Hospital Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Patient population and overall clinical outcomes</div>',
        unsafe_allow_html=True
    )

    total_patients = (
        df[patient_col].nunique()
        if patient_col
        else len(df)
    )

    hospital_mortality = mortality_rate(
        mortality_col
    )

    mortality_28d = mortality_rate(
        mortality28_col
    )

    avg_age = safe_mean(age_col)
    avg_bmi = safe_mean(bmi_col)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "👥",
            "Total Patients",
            f"{total_patients:,}"
        )

    with c2:
        metric_card(
            "⚠️",
            "In-Hospital Mortality",
            display_rate(hospital_mortality)
        )

    with c3:
        metric_card(
            "📅",
            "28-Day Mortality",
            display_rate(mortality_28d)
        )

    with c4:
        metric_card(
            "🎂",
            "Average Age",
            f"{avg_age:.1f}"
            if not pd.isna(avg_age)
            else "N/A"
        )

    st.write("")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        metric_card(
            "⚖️",
            "Average BMI",
            f"{avg_bmi:.1f}"
            if not pd.isna(avg_bmi)
            else "N/A"
        )

    with c6:
        metric_card(
            "❤️",
            "NYHA Available",
            "Yes" if nyha_col else "No"
        )

    with c7:
        metric_card(
            "🚨",
            "Killip Available",
            "Yes" if killip_col else "No"
        )

    with c8:
        metric_card(
            "🧪",
            "Biomarkers Available",
            "Yes"
            if any([
                crp_col,
                wbc_col,
                nlr_col,
                albumin_col
            ])
            else "No"
        )

    st.write("")

    st.markdown(
        """
        <div class="info-box">

        <b>Clinical analytics objective</b><br><br>

        This dashboard evaluates demographic, cardiac,
        inflammatory, nutritional and clinical characteristics
        in relation to mortality outcomes.

        It provides descriptive analysis, mortality association
        analysis, an experimental Artificial Neural Network
        prediction interface, model performance evaluation,
        and data-driven analytical flags.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Outcome Overview")

    outcome_data = []

    if mortality_col:

        rate = mortality_rate(mortality_col)

        outcome_data.append({
            "Outcome": "In-Hospital Mortality",
            "Rate": rate
        })

    if mortality28_col:

        rate = mortality_rate(mortality28_col)

        outcome_data.append({
            "Outcome": "28-Day Mortality",
            "Rate": rate
        })

    if outcome_data:

        outcome_df = pd.DataFrame(
            outcome_data
        )

        fig = px.bar(
            outcome_df,
            x="Outcome",
            y="Rate",
            text="Rate",
            title="Mortality Outcomes"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(
            title="Mortality (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 2. DATA QUALITY
# ============================================================

elif page == "📊 Data Quality":

    st.markdown(
        '<div class="main-title">Data Quality & Structure</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Dataset completeness, duplicates and variable structure</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Columns",
            f"{df.shape[1]:,}"
        )

    with c3:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    with c4:

        missing_pct = (
            df.isna().mean().mean() * 100
        )

        st.metric(
            "Missing Cells",
            f"{missing_pct:.1f}%"
        )

    st.subheader("Column Inventory")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Missing": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Missing %": [
            round(
                df[c].isna().mean() * 100,
                1
            )
            for c in df.columns
        ],
        "Unique Values": [
            df[c].nunique(dropna=True)
            for c in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Missing Values")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) > 0:

        missing_df = (
            missing
            .reset_index()
        )

        missing_df.columns = [
            "Column",
            "Missing Values"
        ]

        fig = px.bar(
            missing_df.head(20),
            x="Missing Values",
            y="Column",
            orientation="h",
            title="Top Missing Variables"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success(
            "No missing values detected."
        )


# ============================================================
# 3. PATIENT PROFILE
# ============================================================

elif page == "👤 Patient Profile":

    st.markdown(
        '<div class="main-title">Patient Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Individual demographic, cardiac, laboratory and outcome profile</div>',
        unsafe_allow_html=True
    )

    if patient_col is None:

        st.warning(
            "Patient identifier column was not detected."
        )

    else:

        patient_values = (
            df[patient_col]
            .dropna()
            .unique()
        )

        patient = st.selectbox(
            "Select Patient",
            patient_values
        )

        patient_df = df[
            df[patient_col] == patient
        ]

        st.subheader("Demographics")

        demographic_data = {}

        for label, col in [
            ("Age", age_col),
            ("Gender", gender_col),
            ("Weight", weight_col),
            ("Height", height_col),
            ("BMI", bmi_col),
            ("Occupation", occupation_col),
            ("Age Category", agecat_col)
        ]:

            if col:

                value = patient_df[col].iloc[0]

                demographic_data[label] = value

        if demographic_data:

            cols = st.columns(
                min(4, len(demographic_data))
            )

            for i, (label, value) in enumerate(
                demographic_data.items()
            ):

                cols[
                    i % len(cols)
                ].metric(
                    label,
                    str(value)
                )

        st.subheader("Cardiac Profile")

        cardiac_data = {}

        for label, col in [
            ("NYHA", nyha_col),
            ("Killip", killip_col)
        ]:

            if col:

                cardiac_data[label] = (
                    patient_df[col].iloc[0]
                )

        if cardiac_data:

            cols = st.columns(
                len(cardiac_data)
            )

            for i, (label, value) in enumerate(
                cardiac_data.items()
            ):

                cols[i].metric(
                    label,
                    str(value)
                )

        st.subheader("Laboratory Profile")

        lab_data = {}

        for label, col in [
            ("hs-CRP", crp_col),
            ("WBC", wbc_col),
            ("NLR", nlr_col),
            ("Albumin", albumin_col)
        ]:

            if col:

                lab_data[label] = (
                    patient_df[col].iloc[0]
                )

        if lab_data:

            cols = st.columns(
                min(4, len(lab_data))
            )

            for i, (label, value) in enumerate(
                lab_data.items()
            ):

                cols[
                    i % len(cols)
                ].metric(
                    label,
                    str(value)
                )

        st.subheader("Mortality Outcomes")

        outcome_data = {}

        if mortality_col:

            outcome_data[
                "In-Hospital Mortality"
            ] = patient_df[
                mortality_col
            ].iloc[0]

        if mortality28_col:

            outcome_data[
                "28-Day Mortality"
            ] = patient_df[
                mortality28_col
            ].iloc[0]

        if outcome_data:

            st.dataframe(
                pd.DataFrame(
                    [
                        outcome_data
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 4. CARDIAC RISK
# ============================================================

elif page == "❤️ Cardiac Risk":

    st.markdown(
        '<div class="main-title">Cardiac Severity & Risk Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">NYHA, Killip and combined cardiac severity analysis</div>',
        unsafe_allow_html=True
    )

    outcome_options = {}

    if mortality_col:
        outcome_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        outcome_options[
            "28-Day Mortality"
        ] = mortality28_col

    if not outcome_options:

        st.error(
            "No mortality outcome column detected."
        )
        st.stop()

    selected_outcome = st.selectbox(
        "Select Outcome",
        list(outcome_options.keys())
    )

    target_col = outcome_options[
        selected_outcome
    ]

    col1, col2 = st.columns(2)

    with col1:

        if nyha_col:

            nyha_data = group_mortality(
                df,
                nyha_col,
                target_col
            )

            if not nyha_data.empty:

                fig = px.bar(
                    nyha_data,
                    x=nyha_col,
                    y="Mortality_Rate",
                    text="Mortality_Rate",
                    title="Mortality by NYHA"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    nyha_data,
                    use_container_width=True,
                    hide_index=True
                )

    with col2:

        if killip_col:

            killip_data = group_mortality(
                df,
                killip_col,
                target_col
            )

            if not killip_data.empty:

                fig = px.bar(
                    killip_data,
                    x=killip_col,
                    y="Mortality_Rate",
                    text="Mortality_Rate",
                    title="Mortality by Killip Grade"
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    killip_data,
                    use_container_width=True,
                    hide_index=True
                )

    # --------------------------------------------------------
    # NYHA x Killip
    # --------------------------------------------------------

    if nyha_col and killip_col:

        st.subheader(
            "NYHA × Killip Mortality Heatmap"
        )

        temp = df[
            [
                nyha_col,
                killip_col,
                target_col
            ]
        ].copy()

        temp["target_binary"] = binary_target(
            temp[target_col]
        )

        temp = temp.dropna(
            subset=[
                nyha_col,
                killip_col,
                "target_binary"
            ]
        )

        if not temp.empty:

            pivot = pd.pivot_table(
                temp,
                values="target_binary",
                index=nyha_col,
                columns=killip_col,
                aggfunc="mean"
            ) * 100

            fig = px.imshow(
                pivot,
                text_auto=".1f",
                aspect="auto",
                title="Mortality Rate (%) by NYHA and Killip"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 5. BIOMARKERS & NUTRITION
# ============================================================

elif page == "🧪 Biomarkers & Nutrition":

    st.markdown(
        '<div class="main-title">Biomarkers & Nutritional Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Inflammation, nutritional status and mortality patterns</div>',
        unsafe_allow_html=True
    )

    outcome_options = {}

    if mortality_col:
        outcome_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        outcome_options[
            "28-Day Mortality"
        ] = mortality28_col

    if not outcome_options:

        st.error(
            "No mortality outcome detected."
        )
        st.stop()

    selected_outcome = st.selectbox(
        "Outcome",
        list(outcome_options.keys())
    )

    target_col = outcome_options[
        selected_outcome
    ]

    biomarker_columns = [
        ("hs-CRP", crp_col),
        ("WBC", wbc_col),
        ("NLR", nlr_col),
        ("Albumin", albumin_col)
    ]

    available = [
        x for x in biomarker_columns
        if x[1] is not None
    ]

    if not available:

        st.warning(
            "No biomarker columns were detected."
        )

    else:

        selected = st.selectbox(
            "Select Biomarker",
            [
                x[0]
                for x in available
            ]
        )

        selected_col = dict(
            available
        )[selected]

        numeric_values = pd.to_numeric(
            df[selected_col],
            errors="coerce"
        )

        temp = pd.DataFrame({
            "Value": numeric_values
        }).dropna()

        fig = px.histogram(
            temp,
            x="Value",
            nbins=30,
            title=f"{selected} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Biomarker vs mortality
        # ----------------------------------------------------

        analysis = df[
            [
                selected_col,
                target_col
            ]
        ].copy()

        analysis["Biomarker"] = pd.to_numeric(
            analysis[selected_col],
            errors="coerce"
        )

        analysis["Mortality"] = binary_target(
            analysis[target_col]
        )

        analysis = analysis.dropna(
            subset=[
                "Biomarker",
                "Mortality"
            ]
        )

        if len(analysis) > 5:

            fig = px.box(
                analysis,
                x="Mortality",
                y="Biomarker",
                points="outliers",
                title=f"{selected} by Mortality Outcome"
            )

            fig.update_xaxes(
                tickvals=[0, 1],
                ticktext=[
                    "Survival",
                    "Mortality"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # Inflammation + Albumin
    # --------------------------------------------------------

    st.subheader(
        "Inflammation + Albumin Risk Groups"
    )

    if crp_col and albumin_col and target_col:

        combo = df[
            [
                crp_col,
                albumin_col,
                target_col
            ]
        ].copy()

        combo["CRP"] = pd.to_numeric(
            combo[crp_col],
            errors="coerce"
        )

        combo["Albumin"] = pd.to_numeric(
            combo[albumin_col],
            errors="coerce"
        )

        combo["Mortality"] = binary_target(
            combo[target_col]
        )

        combo = combo.dropna(
            subset=[
                "CRP",
                "Albumin",
                "Mortality"
            ]
        )

        if len(combo) >= 10:

            crp_cut = combo["CRP"].median()
            albumin_cut = combo["Albumin"].median()

            combo["Inflammation Group"] = np.where(
                combo["CRP"] >= crp_cut,
                "Higher CRP",
                "Lower CRP"
            )

            combo["Albumin Group"] = np.where(
                combo["Albumin"] < albumin_cut,
                "Lower Albumin",
                "Higher Albumin"
            )

            combo["Risk Group"] = (
                combo["Inflammation Group"]
                + " + "
                + combo["Albumin Group"]
            )

            combo_result = (
                combo
                .groupby("Risk Group")["Mortality"]
                .agg(
                    Patients="count",
                    Mortality_Rate="mean"
                )
                .reset_index()
            )

            combo_result[
                "Mortality_Rate"
            ] *= 100

            fig = px.bar(
                combo_result,
                x="Risk Group",
                y="Mortality_Rate",
                text="Mortality_Rate",
                title="Mortality by Inflammation + Albumin Group"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                combo_result,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 6. MORTALITY ANALYSIS
# ============================================================

elif page == "⚠️ Mortality Analysis":

    st.markdown(
        '<div class="main-title">Mortality Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Interactive mortality analysis across demographic and clinical characteristics</div>',
        unsafe_allow_html=True
    )

    outcome_options = {}

    if mortality_col:
        outcome_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        outcome_options[
            "28-Day Mortality"
        ] = mortality28_col

    if not outcome_options:

        st.error(
            "Mortality variables were not detected."
        )
        st.stop()

    selected_outcome = st.selectbox(
        "Select Mortality Outcome",
        list(outcome_options.keys())
    )

    target_col = outcome_options[
        selected_outcome
    ]

    candidate_groups = [
        ("Age", age_col),
        ("Gender", gender_col),
        ("BMI", bmi_col),
        ("Age Category", agecat_col),
        ("NYHA", nyha_col),
        ("Killip", killip_col),
        ("Occupation", occupation_col)
    ]

    available_groups = [
        x for x in candidate_groups
        if x[1] is not None
    ]

    selected_group_name = st.selectbox(
        "Analyze Mortality By",
        [
            x[0]
            for x in available_groups
        ]
    )

    selected_group_col = dict(
        available_groups
    )[selected_group_name]

    result = group_mortality(
        df,
        selected_group_col,
        target_col
    )

    if not result.empty:

        fig = px.bar(
            result,
            x=selected_group_col,
            y="Mortality_Rate",
            text="Mortality_Rate",
            title=f"{selected_outcome} by {selected_group_name}"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_yaxes(
            title="Mortality (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Age vs mortality
    # --------------------------------------------------------

    if age_col:

        st.subheader(
            "Age and Mortality"
        )

        age_data = df[
            [
                age_col,
                target_col
            ]
        ].copy()

        age_data["Age"] = pd.to_numeric(
            age_data[age_col],
            errors="coerce"
        )

        age_data["Mortality"] = binary_target(
            age_data[target_col]
        )

        age_data = age_data.dropna()

        if len(age_data) > 10:

            age_data["Age Group"] = pd.cut(
                age_data["Age"],
                bins=[
                    0,
                    40,
                    50,
                    60,
                    70,
                    80,
                    200
                ],
                labels=[
                    "<40",
                    "40-49",
                    "50-59",
                    "60-69",
                    "70-79",
                    "80+"
                ]
            )

            age_result = (
                age_data
                .groupby(
                    "Age Group",
                    observed=False
                )["Mortality"]
                .agg(
                    Patients="count",
                    Mortality_Rate="mean"
                )
                .reset_index()
            )

            age_result[
                "Mortality_Rate"
            ] *= 100

            fig = px.bar(
                age_result,
                x="Age Group",
                y="Mortality_Rate",
                text="Mortality_Rate",
                title="Mortality by Age Group"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 7. RELATIONSHIP EXPLORER
# ============================================================

elif page == "🔎 Relationship Explorer":

    st.markdown(
        '<div class="main-title">Relationship Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore relationships among demographic, cardiac and laboratory variables</div>',
        unsafe_allow_html=True
    )

    numeric_columns = []

    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().sum() > 10:

            numeric_columns.append(col)

    if len(numeric_columns) >= 2:

        c1, c2 = st.columns(2)

        with c1:

            x_col = st.selectbox(
                "X Variable",
                numeric_columns,
                index=0
            )

        with c2:

            y_col = st.selectbox(
                "Y Variable",
                numeric_columns,
                index=min(
                    1,
                    len(numeric_columns) - 1
                )
            )

        relationship = df[
            [
                x_col,
                y_col
            ]
        ].copy()

        relationship[x_col] = pd.to_numeric(
            relationship[x_col],
            errors="coerce"
        )

        relationship[y_col] = pd.to_numeric(
            relationship[y_col],
            errors="coerce"
        )

        relationship = relationship.dropna()

        if len(relationship) > 2:

            fig = px.scatter(
                relationship,
                x=x_col,
                y=y_col,
                trendline="ols",
                title=f"{x_col} vs {y_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            correlation = (
                relationship[
                    [x_col, y_col]
                ]
                .corr()
                .iloc[0, 1]
            )

            st.metric(
                "Pearson Correlation",
                f"{correlation:.3f}"
            )

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    st.subheader(
        "Clinical Variable Correlation"
    )

    selected_numeric = numeric_columns[
        :min(15, len(numeric_columns))
    ]

    if len(selected_numeric) >= 2:

        corr = (
            df[selected_numeric]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .corr()
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL HELPER
# ============================================================

def prepare_ann_model(target_col):

    # Candidate predictors
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
        albumin_col
    ]

    features = []

    for col in candidate_features:

        if col is not None and col not in features:
            features.append(col)

    if not features:
        return None

    model_df = df[
        features + [target_col]
    ].copy()

    model_df["TARGET"] = binary_target(
        model_df[target_col]
    )

    model_df = model_df.dropna(
        subset=["TARGET"]
    )

    if len(model_df) < 30:
        return None

    y = model_df["TARGET"].astype(int)

    if y.nunique() < 2:
        return None

    X = model_df[
        features
    ].copy()

    categorical_features = []

    numerical_features = []

    for col in features:

        numeric_version = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        numeric_count = (
            numeric_version.notna().sum()
        )

        if numeric_count >= 0.8 * len(X):

            X[col] = numeric_version

            numerical_features.append(col)

        else:

            categorical_features.append(col)

    transformers = []

    if numerical_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    return {
        "pipeline": pipeline,
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "model_df": model_df
    }


# ============================================================
# 8. AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.markdown(
        '<div class="main-title">Artificial Neural Network Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Experimental mortality prediction using demographic, cardiac and laboratory characteristics</div>',
        unsafe_allow_html=True
    )

    outcome_options = {}

    if mortality_col:
        outcome_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        outcome_options[
            "28-Day Mortality"
        ] = mortality28_col

    if not outcome_options:

        st.error(
            "No mortality outcome detected."
        )
        st.stop()

    selected_outcome = st.selectbox(
        "Prediction Target",
        list(outcome_options.keys())
    )

    target_col = outcome_options[
        selected_outcome
    ]

    result = prepare_ann_model(
        target_col
    )

    if result is None:

        st.error(
            "The ANN could not be trained. "
            "Check the target variable and available predictors."
        )

    else:

        model = result["pipeline"]
        features = result["features"]

        st.subheader(
            "Patient Input"
        )

        input_values = {}

        cols = st.columns(2)

        for i, feature in enumerate(features):

            with cols[
                i % 2
            ]:

                numeric_version = pd.to_numeric(
                    df[feature],
                    errors="coerce"
                )

                numeric_count = (
                    numeric_version.notna().sum()
                )

                if numeric_count >= 0.8 * len(df):

                    median_value = (
                        numeric_version
                        .median()
                    )

                    minimum = (
                        numeric_version
                        .min()
                    )

                    maximum = (
                        numeric_version
                        .max()
                    )

                    if pd.isna(median_value):
                        median_value = 0

                    if pd.isna(minimum):
                        minimum = 0

                    if pd.isna(maximum):
                        maximum = median_value + 1

                    if minimum == maximum:
                        maximum = minimum + 1

                    input_values[
                        feature
                    ] = st.number_input(
                        feature.replace(
                            "_",
                            " "
                        ).title(),
                        min_value=float(
                            minimum
                        ),
                        max_value=float(
                            maximum
                        ),
                        value=float(
                            median_value
                        )
                    )

                else:

                    choices = (
                        df[feature]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if choices:

                        input_values[
                            feature
                        ] = st.selectbox(
                            feature.replace(
                                "_",
                                " "
                            ).title(),
                            choices
                        )

        st.write("")

        if st.button(
            "❤️ Calculate Mortality Risk",
            type="primary",
            use_container_width=True
        ):

            patient_input = pd.DataFrame(
                [input_values]
            )

            probability = model.predict_proba(
                patient_input
            )[0, 1]

            prediction = model.predict(
                patient_input
            )[0]

            st.divider()

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Predicted Mortality Probability",
                    f"{probability * 100:.1f}%"
                )

                st.progress(
                    float(probability)
                )

            with c2:

                if prediction == 1:

                    st.markdown(
                        f"""
                        <div class="risk-box">

                        <h3>⚠️ Model Classification</h3>

                        The ANN classified this input
                        into the mortality class.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="insight-box">

                        <h3>✓ Model Classification</h3>

                        The ANN classified this input
                        into the non-mortality class.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.warning(
                "Research-use prediction only. "
                "This model is not a clinical diagnosis "
                "or treatment recommendation."
            )


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-title">ANN Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Evaluation of mortality classification performance</div>',
        unsafe_allow_html=True
    )

    outcome_options = {}

    if mortality_col:
        outcome_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        outcome_options[
            "28-Day Mortality"
        ] = mortality28_col

    if not outcome_options:

        st.error(
            "No mortality outcome detected."
        )
        st.stop()

    selected_outcome = st.selectbox(
        "Model Target",
        list(outcome_options.keys())
    )

    target_col = outcome_options[
        selected_outcome
    ]

    result = prepare_ann_model(
        target_col
    )

    if result is None:

        st.error(
            "Unable to train ANN."
        )

    else:

        y_test = result["y_test"]
        predictions = result["predictions"]
        probabilities = result["probabilities"]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card(
                "🎯",
                "Accuracy",
                f"{accuracy:.3f}"
            )

        with c2:
            metric_card(
                "🔎",
                "Precision",
                f"{precision:.3f}"
            )

        with c3:
            metric_card(
                "🚨",
                "Recall",
                f"{recall:.3f}"
            )

        with c4:
            metric_card(
                "⚖️",
                "F1 Score",
                f"{f1:.3f}"
            )

        with c5:
            metric_card(
                "📈",
                "ROC-AUC",
                f"{auc:.3f}"
            )

        st.divider()

        c1, c2 = st.columns(2)

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        with c1:

            st.subheader(
                "Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                labels={
                    "x": "Predicted",
                    "y": "Actual",
                    "color": "Patients"
                },
                x=[
                    "Survival",
                    "Mortality"
                ],
                y=[
                    "Survival",
                    "Mortality"
                ],
                title="ANN Confusion Matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # ROC
        # ----------------------------------------------------

        with c2:

            st.subheader(
                "ROC Curve"
            )

            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ANN AUC = {auc:.3f}"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random"
                )
            )

            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title="ROC Curve",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader(
            "ANN Input Features"
        )

        feature_table = pd.DataFrame({
            "Feature": result["features"],
            "Used in ANN": "Yes"
        })

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 10. DATA-DRIVEN INSIGHTS
# ============================================================

elif page == "💡 Data-Driven Insights":

    st.markdown(
        '<div class="main-title">Data-Driven Insights & Analytical Flags</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Automatically generated observations from the observed dataset</div>',
        unsafe_allow_html=True
    )

    st.info(
        "These are analytical flags based on observed data patterns. "
        "They are not clinical treatment recommendations."
    )

    target_options = {}

    if mortality_col:
        target_options[
            "In-Hospital Mortality"
        ] = mortality_col

    if mortality28_col:
        target_options[
            "28-Day Mortality"
        ] = mortality28_col

    if target_options:

        selected = st.selectbox(
            "Outcome",
            list(target_options.keys())
        )

        target = target_options[selected]

        # ----------------------------------------------------
        # NYHA flag
        # ----------------------------------------------------

        if nyha_col:

            result = group_mortality(
                df,
                nyha_col,
                target
            )

            if not result.empty:

                highest = result.loc[
                    result["Mortality_Rate"].idxmax()
                ]

                st.markdown(
                    f"""
                    <div class="insight-box">

                    <b>❤️ Cardiac Severity Flag</b><br><br>

                    The observed mortality rate varies across
                    NYHA categories.

                    The category with the highest observed
                    mortality rate in this dataset is:

                    <b>{highest[nyha_col]}</b>

                    with an observed mortality rate of

                    <b>{highest["Mortality_Rate"]:.1f}%</b>.

                    <br><br>

                    <b>Analytical action:</b>
                    review this subgroup alongside other
                    cardiac and laboratory variables.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # Killip flag
        # ----------------------------------------------------

        if killip_col:

            result = group_mortality(
                df,
                killip_col,
                target
            )

            if not result.empty:

                highest = result.loc[
                    result["Mortality_Rate"].idxmax()
                ]

                st.markdown(
                    f"""
                    <div class="insight-box">

                    <b>🚨 Killip Severity Flag</b><br><br>

                    Mortality varies across Killip categories.

                    The category with the highest observed
                    mortality rate is:

                    <b>{highest[killip_col]}</b>

                    with an observed mortality rate of

                    <b>{highest["Mortality_Rate"]:.1f}%</b>.

                    <br><br>

                    <b>Analytical action:</b>
                    examine this subgroup with NYHA,
                    biomarker and demographic characteristics.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # Albumin + inflammation
        # ----------------------------------------------------

        if (
            crp_col
            and albumin_col
        ):

            combo = df[
                [
                    crp_col,
                    albumin_col,
                    target
                ]
            ].copy()

            combo["CRP"] = pd.to_numeric(
                combo[crp_col],
                errors="coerce"
            )

            combo["Albumin"] = pd.to_numeric(
                combo[albumin_col],
                errors="coerce"
            )

            combo["Mortality"] = binary_target(
                combo[target]
            )

            combo = combo.dropna()

            if len(combo) >= 10:

                crp_median = combo[
                    "CRP"
                ].median()

                albumin_median = combo[
                    "Albumin"
                ].median()

                combo["Group"] = np.select(
                    [
                        (
                            combo["CRP"]
                            >= crp_median
                        )
                        &
                        (
                            combo["Albumin"]
                            < albumin_median
                        ),

                        (
                            combo["CRP"]
                            >= crp_median
                        )
                        &
                        (
                            combo["Albumin"]
                            >= albumin_median
                        ),

                        (
                            combo["CRP"]
                            < crp_median
                        )
                        &
                        (
                            combo["Albumin"]
                            < albumin_median
                        )
                    ],
                    [
                        "Higher inflammation + Lower albumin",
                        "Higher inflammation + Higher albumin",
                        "Lower inflammation + Lower albumin"
                    ],
                    default="Lower inflammation + Higher albumin"
                )

                result = (
                    combo
                    .groupby("Group")[
                        "Mortality"
                    ]
                    .agg(
                        Patients="count",
                        Mortality_Rate="mean"
                    )
                    .reset_index()
                )

                result[
                    "Mortality_Rate"
                ] *= 100

                highest = result.loc[
                    result["Mortality_Rate"].idxmax()
                ]

                st.markdown(
                    f"""
                    <div class="warning-box">

                    <b>🧪 Inflammation + Nutrition Flag</b><br><br>

                    The observed mortality rate differs across
                    inflammation and albumin groups.

                    The group with the highest observed mortality
                    rate is:

                    <b>{highest["Group"]}</b>

                    with an observed mortality rate of

                    <b>{highest["Mortality_Rate"]:.1f}%</b>.

                    <br><br>

                    <b>Analytical action:</b>
                    use the combined inflammatory and nutritional
                    profile as a subgroup for further outcome review.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# 11. RESEARCH QUESTIONS
# ============================================================

elif page == "📋 Research Questions":

    st.markdown(
        '<div class="main-title">Research Question Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Connect dashboard analyses with the study questions</div>',
        unsafe_allow_html=True
    )

    questions = {

        "Q1 — Demographics and mortality":
        """
        Can demographic characteristics such as age, gender,
        weight, height, BMI and occupation predict mortality?
        """,

        "Q2 — NYHA and mortality":
        """
        Can NYHA cardiac functional class predict
        in-hospital and 28-day mortality?
        """,

        "Q3 — Killip and mortality":
        """
        Can Killip grade predict short-term mortality?
        """,

        "Q4 — NYHA + Killip":
        """
        Can combined NYHA and Killip severity provide
        additional mortality discrimination?
        """,

        "Q5 — Inflammatory biomarkers":
        """
        Can hs-CRP, WBC and NLR identify mortality patterns?
        """,

        "Q6 — Inflammation + albumin":
        """
        Do elevated inflammatory markers together with
        lower albumin identify a higher-mortality subgroup?
        """,

        "Q7 — Combined clinical model":
        """
        Does adding cardiac and laboratory characteristics
        improve mortality prediction beyond demographic
        characteristics?
        """,

        "Q8 — Artificial Neural Network":
        """
        Can an ANN combine demographic, cardiac and laboratory
        variables to classify mortality outcomes?
        """,

        "Q9 — 28-day mortality":
        """
        Can demographic, clinical, cardiac and laboratory
        characteristics be combined to predict 28-day mortality?
        """
    }

    selected_question = st.selectbox(
        "Select Research Question",
        list(questions.keys())
    )

    st.markdown(
        f"""
        <div class="info-box">

        <h3>{selected_question}</h3>

        {questions[selected_question]}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "Recommended Dashboard Analysis"
    )

    if selected_question.startswith("Q1"):

        st.write(
            "Use Patient Profile, Mortality Analysis and "
            "Relationship Explorer."
        )

    elif selected_question.startswith("Q2"):

        st.write(
            "Use Cardiac Risk → NYHA mortality analysis."
        )

    elif selected_question.startswith("Q3"):

        st.write(
            "Use Cardiac Risk → Killip mortality analysis."
        )

    elif selected_question.startswith("Q4"):

        st.write(
            "Use the NYHA × Killip mortality heatmap."
        )

    elif selected_question.startswith("Q5"):

        st.write(
            "Use Biomarkers & Nutrition and Relationship Explorer."
        )

    elif selected_question.startswith("Q6"):

        st.write(
            "Use the Inflammation + Albumin subgroup analysis."
        )

    elif selected_question.startswith("Q7"):

        st.write(
            "Use AI Prediction and Model Performance."
        )

    elif selected_question.startswith("Q8"):

        st.write(
            "Use AI Prediction and Model Performance."
        )

    elif selected_question.startswith("Q9"):

        st.write(
            "Use AI Prediction with the 28-day mortality target."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Heart Failure Clinical Analytics Dashboard | "
    "Descriptive • Predictive • Data-Driven Analytics"
)

st.caption(
    "For research and analytical use only. "
    "Model outputs should not be interpreted as clinical diagnosis "
    "or treatment recommendations."
)
