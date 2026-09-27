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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #F4F9FB;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #073B4C 0%,
            #0B5D6B 55%,
            #087F5B 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Main title */
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #073B4C;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 16px;
        color: #52727D;
        margin-bottom: 25px;
    }

    /* Hospital header */
    .hospital-header {
        background: linear-gradient(
            90deg,
            #073B4C,
            #087F5B
        );
        padding: 22px 28px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
    }

    .hospital-header h1 {
        margin: 0;
        font-size: 30px;
    }

    .hospital-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
    }

    /* KPI cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #087F5B;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.06);
        min-height: 120px;
    }

    .metric-icon {
        font-size: 28px;
    }

    .metric-title {
        color: #637B83;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #073B4C;
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Section cards */
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Risk box */
    .risk-high {
        background: #FFF1F0;
        border-left: 6px solid #D9534F;
        padding: 20px;
        border-radius: 10px;
    }

    .risk-low {
        background: #EAF8F1;
        border-left: 6px solid #087F5B;
        padding: 20px;
        border-radius: 10px;
    }

    /* Info */
    .info-box {
        background: #EAF5F8;
        border-left: 5px solid #087F9B;
        padding: 15px;
        border-radius: 8px;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA FROM GITHUB REPOSITORY
# ============================================================

from pathlib import Path

DATA_FILE = Path(__file__).parent / "Cardiac_Cleaned_Data.xlsb"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_FILE, engine="pyxlsb")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load Cardiac_Cleaned_Data.xlsb: {e}")
    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(possible_names):

    lower_columns = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:

            return lower_columns[name.lower()]

    return None


age_col = find_column([
    "age",
    "age_years"
])

agecat_col = find_column([
    "agecat",
    "age_category"
])

gender_col = find_column([
    "gender",
    "sex"
])

bmi_col = find_column([
    "bmi",
    "BMI"
])

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

# The cleaned dataset stores in-hospital outcome as text.
outcome_col = find_column([
    "outcome_during_hospitalization"
])

if outcome_col:
    df["in_hospital_mortality"] = (
        df[outcome_col].astype(str).str.strip().str.lower().eq("dead").astype(int)
    )
    mortality_col = "in_hospital_mortality"
else:
    mortality_col = find_column([
        "in_hospital_mortality",
        "hospital_mortality",
        "mortality",
        "death"
    ])

mortality28_col = find_column([
    "mortality_28d",
    "mortality_28_day",
    "28_day_mortality",
    "mortality_28dincrease_28d"
])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        text-align:center;
        padding:15px;
        ">
        <div style="font-size:50px;">❤️</div>
        <h2>HeartCare AI</h2>
        <p style="font-size:13px;">
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
            "👤 Patient Profile",
            "❤️ Cardiac Risk",
            "🧪 Biomarkers",
            "⚠️ Mortality Analysis",
            "🤖 AI Prediction",
            "📈 Model Performance"
        ]
    )

    st.divider()

    st.caption(
        "Clinical analytics dashboard"
    )

    st.caption(
        "For research and analytical use."
    )


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="hospital-header">
        <h1>❤️ Heart Failure Clinical Analytics</h1>
        <p>Patient Risk Stratification • Mortality Analysis • Artificial Neural Network Prediction</p>
    </div>
    """
)


# ============================================================
# HELPER FUNCTION FOR KPI
# ============================================================

def metric_card(icon, title, value):
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    )


# ============================================================
# OVERVIEW
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

    # -------------------------
    # KPIs
    # -------------------------

    total_patients = len(df)

    if mortality_col:

        mortality_rate = (
            pd.to_numeric(
                df[mortality_col],
                errors="coerce"
            ).mean() * 100
        )

    else:

        mortality_rate = np.nan


    avg_age = (
        pd.to_numeric(
            df[age_col],
            errors="coerce"
        ).mean()
        if age_col else np.nan
    )


    avg_bmi = (
        pd.to_numeric(
            df[bmi_col],
            errors="coerce"
        ).mean()
        if bmi_col else np.nan
    )


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
            f"{mortality_rate:.1f}%"
            if not np.isnan(mortality_rate)
            else "N/A"
        )

    with c3:

        metric_card(
            "🎂",
            "Age Data Available",
            f"{df[age_col].notna().sum():,}"
            if age_col
            else (f"{df[agecat_col].notna().sum():,}" if agecat_col else "N/A")
        )

    with c4:

        metric_card(
            "⚖️",
            "Average BMI",
            f"{avg_bmi:.1f}"
            if not np.isnan(avg_bmi)
            else "N/A"
        )


    st.write("")


    # -------------------------
    # Quick summary
    # -------------------------

    st.markdown(
        """
        <div class="info-box">

        <b>Clinical analytics objective</b><br>

        This dashboard evaluates demographic, cardiac,
        inflammatory, nutritional, and clinical characteristics
        in relation to mortality outcomes and provides an
        experimental machine-learning prediction interface.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    # -------------------------
    # Charts
    # -------------------------

    col1, col2 = st.columns(2)


    if mortality_col:

        with col1:

            st.subheader("❤️ Outcome Distribution")

            outcome = (
                df[mortality_col]
                .value_counts()
                .reset_index()
            )

            outcome.columns = [
                "Outcome",
                "Patients"
            ]

            outcome["Outcome"] = (
                outcome["Outcome"]
                .map({
                    0: "Survived",
                    1: "Mortality"
                })
                .fillna(
                    outcome["Outcome"].astype(str)
                )
            )

            fig = px.pie(
                outcome,
                names="Outcome",
                values="Patients",
                hole=0.55
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    if age_col:

        with col2:

            st.subheader("🎂 Age Distribution")

            fig = px.histogram(
                df,
                x=age_col,
                nbins=25,
                marginal="box"
            )

            fig.update_layout(
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# PATIENT PROFILE
# ============================================================

elif page == "👤 Patient Profile":

    st.header("👤 Patient Profile")

    available_profile = [
        c for c in [
            age_col,
            gender_col,
            bmi_col
        ]
        if c is not None
    ]


    if len(available_profile) == 0:

        st.warning(
            "No demographic columns were detected."
        )

    else:

        st.subheader(
            "Patient Characteristics"
        )

        selected_column = st.selectbox(
            "Select characteristic",
            available_profile
        )

        fig = px.histogram(
            df,
            x=selected_column,
            marginal="box"
        )

        fig.update_layout(
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Gender

    if gender_col and mortality_col:

        st.subheader(
            "Gender and Mortality"
        )

        temp = df.copy()

        temp["_mortality"] = pd.to_numeric(
            temp[mortality_col],
            errors="coerce"
        )

        gender_result = (
            temp.groupby(gender_col)["_mortality"]
            .mean()
            .reset_index()
        )

        gender_result["_mortality"] *= 100

        fig = px.bar(
            gender_result,
            x=gender_col,
            y="_mortality",
            text="_mortality"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%"
        )

        fig.update_yaxes(
            title="Mortality (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CARDIAC RISK
# ============================================================

elif page == "❤️ Cardiac Risk":

    st.header("❤️ Cardiac Severity Analysis")

    st.markdown(
        "Evaluate mortality patterns across NYHA and Killip severity categories."
    )


    col1, col2 = st.columns(2)


    # NYHA

    if nyha_col and mortality_col:

        with col1:

            st.subheader(
                "NYHA Functional Class"
            )

            temp = df.copy()

            temp["_mortality"] = pd.to_numeric(
                temp[mortality_col],
                errors="coerce"
            )

            result = (
                temp.groupby(nyha_col)["_mortality"]
                .mean()
                .reset_index()
            )

            result["_mortality"] *= 100

            fig = px.bar(
                result,
                x=nyha_col,
                y="_mortality",
                text="_mortality"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%"
            )

            fig.update_yaxes(
                title="Mortality (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # Killip

    if killip_col and mortality_col:

        with col2:

            st.subheader(
                "Killip Grade"
            )

            temp = df.copy()

            temp["_mortality"] = pd.to_numeric(
                temp[mortality_col],
                errors="coerce"
            )

            result = (
                temp.groupby(killip_col)["_mortality"]
                .mean()
                .reset_index()
            )

            result["_mortality"] *= 100

            fig = px.bar(
                result,
                x=killip_col,
                y="_mortality",
                text="_mortality"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%"
            )

            fig.update_yaxes(
                title="Mortality (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # NYHA x Killip

    if (
        nyha_col
        and killip_col
        and mortality_col
    ):

        st.subheader(
            "NYHA × Killip Mortality Map"
        )

        temp = df.copy()

        temp["_mortality"] = pd.to_numeric(
            temp[mortality_col],
            errors="coerce"
        )

        pivot = temp.pivot_table(
            values="_mortality",
            index=nyha_col,
            columns=killip_col,
            aggfunc="mean"
        ) * 100


        fig = px.imshow(
            pivot,
            text_auto=".1f",
            aspect="auto",
            labels={
                "color": "Mortality (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# BIOMARKERS
# ============================================================

elif page == "🧪 Biomarkers":

    st.header("🧪 Biomarker & Nutrition Analysis")

    biomarker_columns = [
        c for c in [
            crp_col,
            wbc_col,
            nlr_col,
            albumin_col
        ]
        if c is not None
    ]


    if len(biomarker_columns) == 0:

        st.warning(
            "No biomarker columns were detected."
        )

    else:

        selected = st.selectbox(
            "Select biomarker",
            biomarker_columns
        )


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "Distribution"
            )

            fig = px.histogram(
                df,
                x=selected,
                nbins=30,
                marginal="box"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            if mortality_col:

                st.subheader(
                    "Biomarker vs Mortality"
                )

                temp = df[
                    [selected, mortality_col]
                ].copy()

                temp[selected] = pd.to_numeric(
                    temp[selected],
                    errors="coerce"
                )

                temp[mortality_col] = pd.to_numeric(
                    temp[mortality_col],
                    errors="coerce"
                )

                temp = temp.dropna()


                if len(temp) > 0:

                    temp["Biomarker Group"] = pd.qcut(
                        temp[selected],
                        q=3,
                        duplicates="drop"
                    )

                    result = (
                        temp.groupby(
                            "Biomarker Group",
                            observed=False
                        )[mortality_col]
                        .mean()
                        .reset_index()
                    )

                    result[mortality_col] *= 100

                    fig = px.bar(
                        result,
                        x="Biomarker Group",
                        y=mortality_col,
                        text=mortality_col
                    )

                    fig.update_traces(
                        texttemplate="%{text:.1f}%"
                    )

                    fig.update_yaxes(
                        title="Mortality (%)"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


    # Biomarker table

    st.subheader(
        "🧪 Biomarker Summary"
    )

    if biomarker_columns:

        summary = df[
            biomarker_columns
        ].describe().T

        summary = summary[
            [
                "count",
                "mean",
                "std",
                "min",
                "50%",
                "max"
            ]
        ]

        st.dataframe(
            summary,
            use_container_width=True
        )


# ============================================================
# MORTALITY ANALYSIS
# ============================================================

elif page == "⚠️ Mortality Analysis":

    st.header(
        "⚠️ Mortality Risk Analysis"
    )


    if not mortality_col:

        st.warning(
            "Mortality column was not detected."
        )

        st.stop()


    # Filters

    st.subheader(
        "🔎 Interactive Filters"
    )


    filtered = df.copy()


    if age_col:

        min_age = int(
            pd.to_numeric(
                df[age_col],
                errors="coerce"
            ).min()
        )

        max_age = int(
            pd.to_numeric(
                df[age_col],
                errors="coerce"
            ).max()
        )


        age_range = st.slider(
            "Age range",
            min_age,
            max_age,
            (min_age, max_age)
        )


        filtered = filtered[
            pd.to_numeric(
                filtered[age_col],
                errors="coerce"
            ).between(
                age_range[0],
                age_range[1]
            )
        ]


    if nyha_col:

        nyha_values = sorted(
            df[nyha_col]
            .dropna()
            .unique()
            .tolist()
        )

        selected_nyha = st.multiselect(
            "NYHA class",
            nyha_values,
            default=nyha_values
        )

        if selected_nyha:

            filtered = filtered[
                filtered[nyha_col].isin(
                    selected_nyha
                )
            ]


    st.write(
        f"Showing **{len(filtered):,}** patients."
    )


    mortality_rate = (
        pd.to_numeric(
            filtered[mortality_col],
            errors="coerce"
        ).mean() * 100
    )


    st.metric(
        "Filtered Mortality Rate",
        f"{mortality_rate:.1f}%"
    )


    # Download

    csv = filtered.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "📥 Download Filtered Data",
        csv,
        "filtered_heart_failure_data.csv",
        "text/csv"
    )


    st.divider()


    # Mortality table

    st.subheader(
        "Mortality Summary"
    )


    if nyha_col:

        result = (
            filtered.groupby(nyha_col)[
                mortality_col
            ]
            .mean()
            .reset_index()
        )

        result["Mortality (%)"] = (
            result[mortality_col] * 100
        )

        result = result.drop(
            columns=[mortality_col]
        )

        st.dataframe(
            result,
            use_container_width=True
        )


# ============================================================
# AI PREDICTION
# ============================================================

elif page == "🤖 AI Prediction":

    st.header(
        "🤖 Artificial Neural Network Prediction"
    )

    st.markdown(
        """
        <div class="info-box">

        Enter patient characteristics to generate a model-based
        probability estimate for the selected mortality outcome.

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    candidate_features = [
        age_col,
        gender_col,
        bmi_col,
        nyha_col,
        killip_col,
        crp_col,
        wbc_col,
        nlr_col,
        albumin_col
    ]


    features = [
        c for c in candidate_features
        if c is not None
    ]


    if not mortality_col:

        st.error(
            "Mortality outcome column was not detected."
        )

        st.stop()


    if len(features) < 2:

        st.error(
            "Not enough prediction variables were detected."
        )

        st.stop()


    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    model_df = df[
        features + [mortality_col]
    ].copy()


    for c in model_df.columns:

        model_df[c] = pd.to_numeric(
            model_df[c],
            errors="coerce"
        )


    model_df = model_df.dropna()


    if model_df[mortality_col].nunique() < 2:

        st.error(
            "The mortality outcome must contain both 0 and 1."
        )

        st.stop()


    X = model_df[
        features
    ]

    y = model_df[
        mortality_col
    ].astype(int)


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
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
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # PATIENT INPUT
    # --------------------------------------------------------

    st.subheader(
        "👤 Patient Information"
    )


    input_values = {}


    columns = st.columns(3)


    for i, feature in enumerate(features):

        with columns[i % 3]:

            series = pd.to_numeric(
                df[feature],
                errors="coerce"
            ).dropna()


            if len(series) == 0:

                continue


            minimum = float(
                series.min()
            )

            maximum = float(
                series.max()
            )

            median = float(
                series.median()
            )


            if feature == gender_col:

                input_values[feature] = st.selectbox(
                    "Gender",
                    sorted(
                        series.unique().tolist()
                    )
                )

            elif feature == nyha_col:

                input_values[feature] = st.selectbox(
                    "NYHA Class",
                    sorted(
                        series.unique().tolist()
                    )
                )

            elif feature == killip_col:

                input_values[feature] = st.selectbox(
                    "Killip Grade",
                    sorted(
                        series.unique().tolist()
                    )
                )

            else:

                input_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=median
                )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "❤️ Calculate Mortality Risk",
        type="primary",
        use_container_width=True
    ):

        patient = pd.DataFrame(
            [input_values]
        )


        patient = patient[
            features
        ]


        patient_scaled = scaler.transform(
            patient
        )


        probability = model.predict_proba(
            patient_scaled
        )[0, 1]


        prediction = model.predict(
            patient_scaled
        )[0]


        st.divider()


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Mortality Probability",
                f"{probability * 100:.1f}%"
            )


            st.progress(
                float(probability)
            )


        with col2:

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="risk-high">

                    <h3>⚠️ Model Classification</h3>

                    The model classified this patient
                    into the mortality class.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">

                    <h3>✓ Model Classification</h3>

                    The model classified this patient
                    into the non-mortality class.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.warning(
            "This is a research prediction model and not a "
            "clinical diagnosis or treatment recommendation."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 ANN Model Performance"
    )


    candidate_features = [
        age_col,
        gender_col,
        bmi_col,
        nyha_col,
        killip_col,
        crp_col,
        wbc_col,
        nlr_col,
        albumin_col
    ]


    features = [
        c for c in candidate_features
        if c is not None
    ]


    if not mortality_col:

        st.error(
            "Mortality column not detected."
        )

        st.stop()


    model_df = df[
        features + [mortality_col]
    ].copy()


    for c in model_df.columns:

        model_df[c] = pd.to_numeric(
            model_df[c],
            errors="coerce"
        )


    model_df = model_df.dropna()


    X = model_df[features]

    y = model_df[mortality_col].astype(int)


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
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
        random_state=42
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]


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


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

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
            }
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    with col2:

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
            height=450
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🧬 Model Input Features"
    )


    feature_table = pd.DataFrame(
        {
            "Feature": features,
            "Used in ANN": ["Yes"] * len(features)
        }
    )


    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )
