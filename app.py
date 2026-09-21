import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Sustainable Agriculture Advisor",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 20px;
}

.info-card {
    padding: 18px;
    border-radius: 12px;
    background-color: #f5f9f5;
    border: 1px solid #dce8dc;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "seasonal_agriculture_performance_dataset.csv"

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:
    st.error(
        "❌ Dataset not found.\n\n"
        "Make sure 'seasonal_agriculture_performance_dataset.csv' "
        "is in the same folder as app.py."
    )
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Crop",
    "Season",
    "State",
    "Irrigation_Method",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Farm_Area_Hectares",
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing from your dataset:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

numeric_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Farm_Area_Hectares",
    "Yield_Tonnes_Ha",
    "Profit_INR",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=required_columns
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌱 AI-Powered Sustainable Agriculture Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An AI-based decision-support system that analyzes agricultural '
    'data to provide insights for productive and sustainable farming.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Agricultural Records",
        f"{len(df):,}"
    )

with m2:
    st.metric(
        "Dataset Features",
        len(df.columns)
    )

with m3:
    st.metric(
        "Crop Types",
        df["Crop"].nunique()
    )

with m4:
    st.metric(
        "States",
        df["State"].nunique()
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🌾 Farm Information")

crop_options = sorted(
    df["Crop"].dropna().unique()
)

season_options = sorted(
    df["Season"].dropna().unique()
)

state_options = sorted(
    df["State"].dropna().unique()
)

irrigation_options = sorted(
    df["Irrigation_Method"].dropna().unique()
)

crop = st.sidebar.selectbox(
    "Crop",
    crop_options
)

season = st.sidebar.selectbox(
    "Season",
    season_options
)

state = st.sidebar.selectbox(
    "State",
    state_options
)

irrigation = st.sidebar.selectbox(
    "Irrigation Method",
    irrigation_options
)

rainfall = st.sidebar.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=float(
        df["Rainfall_mm"].median()
    )
)

temperature = st.sidebar.number_input(
    "Average Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=float(
        df["Avg_Temperature_C"].median()
    )
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(
        df["Humidity_pct"].median()
    )
)

farm_area = st.sidebar.number_input(
    "Farm Area (hectares)",
    min_value=0.1,
    value=float(
        df["Farm_Area_Hectares"].median()
    )
)

analyze = st.sidebar.button(
    "🔍 Analyze Farm",
    use_container_width=True
)


# ============================================================
# AI MODEL
# ============================================================

model_features = [
    "Crop",
    "Season",
    "State",
    "Irrigation_Method",
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Farm_Area_Hectares"
]

categorical_features = [
    "Crop",
    "Season",
    "State",
    "Irrigation_Method"
]

numerical_features = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Farm_Area_Hectares"
]


X = df[model_features]

y = df["Yield_Tonnes_Ha"]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    max_depth=12,
    min_samples_split=4,
    n_jobs=-1
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


# ============================================================
# TRAIN MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

pipeline.fit(
    X_train,
    y_train
)

predictions = pipeline.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if analyze:

    # --------------------------------------------------------
    # INPUT DATA FOR MODEL
    # --------------------------------------------------------

    user_input = pd.DataFrame({
        "Crop": [crop],
        "Season": [season],
        "State": [state],
        "Irrigation_Method": [irrigation],
        "Rainfall_mm": [rainfall],
        "Avg_Temperature_C": [temperature],
        "Humidity_pct": [humidity],
        "Farm_Area_Hectares": [farm_area]
    })


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_yield = pipeline.predict(
        user_input
    )[0]


    # --------------------------------------------------------
    # HISTORICAL FILTER
    # --------------------------------------------------------

    filtered = df[
        (df["Crop"] == crop) &
        (df["Season"] == season) &
        (df["State"] == state) &
        (df["Irrigation_Method"] == irrigation)
    ]


    # If exact combination doesn't exist
    if len(filtered) == 0:

        filtered = df[
            (df["Crop"] == crop) &
            (df["Season"] == season)
        ]


    # --------------------------------------------------------
    # HISTORICAL STATISTICS
    # --------------------------------------------------------

    avg_yield = filtered[
        "Yield_Tonnes_Ha"
    ].mean()

    avg_profit = filtered[
        "Profit_INR"
    ].mean()

    avg_water = filtered[
        "Water_Used_m3"
    ].mean()

    avg_efficiency = filtered[
        "Water_Efficiency_t_per_1000m3"
    ].mean()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Agricultural Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "AI Predicted Yield",
            f"{predicted_yield:.2f} t/ha"
        )

    with c2:
        st.metric(
            "Historical Yield",
            f"{avg_yield:.2f} t/ha"
        )

    with c3:
        st.metric(
            "Avg Profit",
            f"₹{avg_profit:,.0f}"
        )

    with c4:
        st.metric(
            "Avg Water Used",
            f"{avg_water:,.0f} m³"
        )

    with c5:
        st.metric(
            "Water Efficiency",
            f"{avg_efficiency:.2f}"
        )


    st.divider()


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "🧠 Machine Learning Model Performance"
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Model",
            "Random Forest"
        )

    with p2:
        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )

    with p3:
        st.metric(
            "Mean Absolute Error",
            f"{mae:.3f} t/ha"
        )


    st.caption(
        "The model was trained using historical agricultural records "
        "and evaluated on a held-out test set."
    )


    # ========================================================
    # SUSTAINABILITY SCORE
    # ========================================================

    st.subheader(
        "🌍 Sustainability Analysis"
    )

    median_efficiency = df[
        "Water_Efficiency_t_per_1000m3"
    ].median()

    if avg_efficiency >= median_efficiency * 1.15:

        sustainability_score = 90
        sustainability_status = "High"

    elif avg_efficiency >= median_efficiency:

        sustainability_score = 75
        sustainability_status = "Moderate-High"

    elif avg_efficiency >= median_efficiency * 0.85:

        sustainability_score = 60
        sustainability_status = "Moderate"

    else:

        sustainability_score = 45
        sustainability_status = "Needs Improvement"


    s1, s2 = st.columns(2)

    with s1:

        st.metric(
            "Sustainability Indicator",
            f"{sustainability_score}/100"
        )

    with s2:

        st.metric(
            "Status",
            sustainability_status
        )


    # ========================================================
    # AI RECOMMENDATION
    # ========================================================

    st.subheader(
        "💡 AI-Generated Farming Insights"
    )


    recommendations = []


    # Yield comparison

    if predicted_yield > avg_yield:

        recommendations.append(
            f"🌾 The model predicts approximately "
            f"{predicted_yield:.2f} tonnes/hectare under the "
            f"selected conditions, which is above the historical "
            f"average of {avg_yield:.2f} tonnes/hectare."
        )

    else:

        recommendations.append(
            f"🌾 The model predicts approximately "
            f"{predicted_yield:.2f} tonnes/hectare under the "
            f"selected conditions. Historical performance should "
            f"be considered when interpreting this estimate."
        )


    # Water analysis

    if avg_efficiency >= median_efficiency:

        recommendations.append(
            "💧 Historical records for the selected conditions "
            "show relatively strong water-use efficiency."
        )

    else:

        recommendations.append(
            "💧 Historical records indicate an opportunity to "
            "improve water-use efficiency. Irrigation scheduling "
            "and local water availability should be considered."
        )


    # Rainfall

    rainfall_median = df[
        "Rainfall_mm"
    ].median()

    if rainfall < rainfall_median:

        recommendations.append(
            "🌦️ The entered rainfall value is below the "
            "dataset median. Water availability should be "
            "monitored before making irrigation decisions."
        )

    else:

        recommendations.append(
            "🌦️ The entered rainfall value is at or above "
            "the dataset median."
        )


    # General recommendation

    recommendations.append(
        "⚠️ These are data-driven decision-support insights, "
        "not guaranteed outcomes. Local weather conditions, "
        "soil conditions and agricultural expertise should also "
        "be considered."
    )


    for recommendation in recommendations:

        st.info(
            recommendation
        )


    # ========================================================
    # CHARTS
    # ========================================================

    st.subheader(
        "📊 Agricultural Data Insights"
    )


    chart1, chart2 = st.columns(2)


    with chart1:

        crop_summary = (
            df.groupby("Crop")[
                "Yield_Tonnes_Ha"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        st.write(
            "**Average Yield by Crop**"
        )

        st.bar_chart(
            crop_summary
        )


    with chart2:

        season_summary = (
            df.groupby("Season")[
                "Yield_Tonnes_Ha"
            ]
            .mean()
        )

        st.write(
            "**Average Yield by Season**"
        )

        st.bar_chart(
            season_summary
        )


    chart3, chart4 = st.columns(2)


    with chart3:

        irrigation_summary = (
            df.groupby("Irrigation_Method")[
                "Yield_Tonnes_Ha"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.write(
            "**Yield by Irrigation Method**"
        )

        st.bar_chart(
            irrigation_summary
        )


    with chart4:

        water_summary = (
            df.groupby("Irrigation_Method")[
                "Water_Efficiency_t_per_1000m3"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        st.write(
            "**Water Efficiency by Irrigation Method**"
        )

        st.bar_chart(
            water_summary
        )


    # ========================================================
    # HISTORICAL RECORDS
    # ========================================================

    st.subheader(
        "📋 Historical Records Used for Context"
    )

    st.write(
        f"Showing up to 20 records relevant to "
        f"{crop} and {season}."
    )

    st.dataframe(
        filtered.head(20),
        use_container_width=True,
        height=300
    )


else:

    # ========================================================
    # WELCOME SCREEN
    # ========================================================

    st.subheader(
        "🌾 How to Use the Advisor"
    )

    st.write(
        """
        Select the agricultural conditions from the sidebar
        and click **Analyze Farm**.

        The system will use historical agricultural data and
        a machine-learning model to generate:

        • AI-based yield prediction  
        • Historical performance comparison  
        • Profit and water-use insights  
        • Water-efficiency analysis  
        • Sustainability indicator  
        • AI-generated recommendations  
        • Agricultural charts
        """
    )


    st.info(
        "👈 Start by selecting your crop, season, state and "
        "irrigation method from the sidebar."
    )


# ============================================================
# RESPONSIBLE AI
# ============================================================

st.divider()

st.subheader(
    "🔐 Responsible AI Considerations"
)

ra1, ra2, ra3, ra4 = st.columns(4)

with ra1:

    st.markdown(
        """
        **⚖️ Fairness**

        Historical data may not represent every farming
        region or crop equally.
        """
    )

with ra2:

    st.markdown(
        """
        **🔎 Transparency**

        Predictions are based on the agricultural features
        used by the model.
        """
    )

with ra3:

    st.markdown(
        """
        **🔒 Privacy**

        The prototype avoids collecting unnecessary
        personal or sensitive farmer information.
        """
    )

with ra4:

    st.markdown(
        """
        **👨‍🌾 Human Oversight**

        AI provides decision support and does not replace
        farmers or agricultural experts.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI for Sustainability Project | 1M1B Virtual Internship | "
    "SDG 2: Zero Hunger"
)