from pathlib import Path
from datetime import datetime
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "dataset" / "student_data.csv"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# YELLOW + WHITE THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --yellow: #FFD400;
        --gold: #EAB308;
        --white: #FFFFFF;
        --muted: #D1D5DB;
        --background: #0D0F14;
        --panel: #191B22;
        --border: rgba(255, 212, 0, 0.35);
    }

    /* Main application background */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #0D0F14 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stHeader"] {
        background: #0D0F14 !important;
    }

    [data-testid="stMain"] {
        background: #0D0F14 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #191B22 !important;
        border-right: 1px solid rgba(255, 212, 0, 0.35);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #D1D5DB !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #FFD400 !important;
        font-weight: 750 !important;
    }

    p, li, label {
        color: #FFFFFF;
    }

    /* Hero section */
    .hero {
        padding: 1.7rem 1.8rem;
        border: 1px solid rgba(255, 212, 0, 0.45);
        border-radius: 20px;
        background: linear-gradient(120deg, #292719, #191B22 80%);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero-kicker {
        color: #FFD400 !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        font-weight: 800;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.25rem;
        margin: 0.45rem 0 0.65rem;
    }

    .hero p {
        color: #F3F4F6 !important;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #191B22 !important;
        border: 1px solid rgba(255, 212, 0, 0.4) !important;
        border-left: 4px solid #FFD400 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #D1D5DB !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #FFD400 !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }

    /* Buttons */
    div.stButton > button,
    div.stDownloadButton > button {
        background: #FFD400 !important;
        color: #111111 !important;
        border: 1px solid #EAB308 !important;
        border-radius: 10px !important;
        font-weight: 750 !important;
        min-height: 2.7rem;
        transition: all 0.2s ease;
    }

    div.stButton > button *,
    div.stDownloadButton > button * {
        color: #111111 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #FFE566 !important;
        border-color: #FFE566 !important;
        transform: translateY(-1px);
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }

    /* Slider values and tracks */
    [data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #FFD400 !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #FFD400 !important;
        border-color: #FFD400 !important;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div {
        background: #343741 !important;
        border-radius: 10px !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #FFD400 !important;
        border-radius: 10px !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #191B22 !important;
        border: 1px solid rgba(255, 212, 0, 0.35) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid rgba(255, 212, 0, 0.3) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        color: #D1D5DB !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 212, 0, 0.25) !important;
    }

    /* Links */
    a {
        color: #FFD400 !important;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero {
            padding: 1rem;
        }

        .hero h1 {
            font-size: 1.6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            Machine Learning • Student Insights
        </div>
        <h1>🎓 Student Performance Predictor</h1>
        <p>
            Explore how study habits and academic indicators relate to an
            estimated exam score. This project uses synthetic sample data
            for learning and demonstration—not high-stakes decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run "
        "`python src/train_model.py` from your project folder."
    )
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# =========================================================
# SIDEBAR INPUTS
# =========================================================

with st.sidebar:
    st.markdown("## 🎓 Student details")
    st.caption("Adjust your inputs and click Predict exam score.")

    study_hours = st.slider(
        "📚 Study hours per day",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
    )

    attendance = st.slider(
        "📅 Attendance (%)",
        min_value=0,
        max_value=100,
        value=85,
        step=1,
    )

    previous_score = st.slider(
        "📝 Previous exam score",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
    )

    assignments_completed = st.slider(
        "✅ Assignments completed (out of 10)",
        min_value=0,
        max_value=10,
        value=7,
        step=1,
    )

    predict_clicked = st.button(
        "🔮 Predict exam score",
        type="primary",
        width="stretch",
    )

    st.divider()

    st.markdown("### 📌 Demo note")
    st.caption(
        "This project uses a synthetically generated dataset. "
        "Predictions are illustrative and are not guaranteed results."
    )


# =========================================================
# PREPARE INPUT AND PREDICT
# =========================================================

input_values = {
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments_completed": assignments_completed,
}

try:
    input_data = pd.DataFrame([input_values])[features]

    predicted_score = float(model.predict(input_data)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()


# =========================================================
# PREDICTION HISTORY (STATE)
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if predict_clicked:
    history_item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "predicted_score": round(predicted_score, 1),
    }

    st.session_state.prediction_history.insert(0, history_item)
    st.session_state.prediction_history = (
        st.session_state.prediction_history[:50]
    )


# =========================================================
# PERFORMANCE CATEGORY AND SUGGESTIONS
# =========================================================

if predicted_score >= 80:
    category = "Strong"
    category_message = (
        "Your inputs fall in the strong estimated performance range."
    )

    suggestions = [
        "Keep a consistent study schedule and take short revision breaks.",
        "Continue completing assignments and review mistakes from practice tests.",
        "Use timed mock exams to practise applying what you have learned.",
    ]

elif predicted_score >= 60:
    category = "Moderate"
    category_message = (
        "Your inputs fall in the moderate estimated performance range."
    )

    suggestions = [
        "Create a regular study plan with short, focused sessions.",
        "Review topics where previous tests were difficult.",
        "Complete assignments on time and revise difficult concepts.",
    ]

else:
    category = "Needs more preparation"
    category_message = (
        "A structured revision plan may help strengthen preparation."
    )

    suggestions = [
        "Start with small daily study blocks and build consistency.",
        "Review foundational concepts and practise sample questions.",
        "Break assignments into smaller tasks and ask for help when needed.",
    ]


# =========================================================
# PREDICTION OVERVIEW
# =========================================================

st.subheader("📌 Prediction overview")

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Estimated exam score",
    f"{predicted_score:.1f} / 100",
)

metric_columns[1].metric(
    "Performance range",
    category,
)

metric_columns[2].metric(
    "Study hours/day",
    f"{study_hours:g} h",
)

metric_columns[3].metric(
    "Attendance",
    f"{attendance}%",
)


# =========================================================
# SCORE ESTIMATE AND SUGGESTIONS
# =========================================================

prediction_column, suggestions_column = st.columns(
    [1.05, 1],
    gap="large",
)

with prediction_column:
    st.markdown("### 📊 Score estimate")

    st.progress(int(round(predicted_score)))

    st.caption(
        "Score scale: 0–100. This is a model estimate, "
        "not a guaranteed exam result."
    )

    if category == "Strong":
        st.success(f"🟢 {category_message}")

    elif category == "Moderate":
        st.info(f"🟡 {category_message}")

    else:
        st.warning(f"🔴 {category_message}")

    # Input profile chart
    chart_data = pd.DataFrame(
        {
            "Indicator": [
                "Study hours (scaled)",
                "Attendance",
                "Previous score",
                "Assignments",
            ],
            "Input value (%)": [
                study_hours / 12 * 100,
                attendance,
                previous_score,
                assignments_completed / 10 * 100,
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(7, 3.5),
        facecolor="#0D0F14",
    )

    ax.set_facecolor("#191B22")

    ax.barh(
        chart_data["Indicator"],
        chart_data["Input value (%)"],
        color="#FFD400",
        edgecolor="#EAB308",
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Input level (%)", color="#FFFFFF")
    ax.set_title(
        "Your input profile",
        color="#FFD400",
        fontweight="bold",
    )

    ax.tick_params(colors="#FFFFFF")
    ax.grid(axis="x", color="#444650", alpha=0.65)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#555967")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


with suggestions_column:
    st.markdown("### 💡 Suggested next steps")

    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")

    st.markdown("### 🧾 Your input summary")

    summary = pd.DataFrame(
        [
            {
                "Study hours/day": study_hours,
                "Attendance (%)": attendance,
                "Previous score": previous_score,
                "Assignments completed": assignments_completed,
            }
        ]
    )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )

    # Download current prediction
    report = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "study_hours_per_day": study_hours,
                "attendance_percent": attendance,
                "previous_exam_score": previous_score,
                "assignments_completed_out_of_10": assignments_completed,
                "predicted_exam_score": round(predicted_score, 1),
                "performance_range": category,
                "note": (
                    "Synthetic-data demo; not for real educational decisions."
                ),
            }
        ]
    )

    st.download_button(
        "⬇️ Download this prediction (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_report.csv",
        mime="text/csv",
        width="stretch",
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.divider()
st.subheader("📈 Model evaluation")

if METRICS_PATH.exists():
    try:
        metrics = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "MAE",
            f"{float(metrics['mae']):.3f}",
            help="Mean Absolute Error; lower is generally better.",
        )

        metric2.metric(
            "RMSE",
            f"{float(metrics['rmse']):.3f}",
            help="Root Mean Squared Error; lower is generally better.",
        )

        metric3.metric(
            "R² score",
            f"{float(metrics['r2']):.3f}",
            help="Closer to 1 indicates a better fit on this test split.",
        )

        st.caption(
            f"Evaluated on {metrics.get('test_rows', 'unknown')} "
            "held-out sample rows. These metrics reflect a synthetic "
            "teaching dataset and do not establish real-world accuracy."
        )

    except (ValueError, KeyError, TypeError) as error:
        st.warning(f"Could not read model evaluation metrics: {error}")

else:
    st.info(
        "Metrics file not found. Run `python src/train_model.py` "
        "to generate it."
    )


# =========================================================
# DATASET EXPLORER
# =========================================================

if DATA_PATH.exists():
    try:
        data = pd.read_csv(DATA_PATH)

        st.divider()
        st.subheader("🔎 Explore the sample dataset")

        data_metric1, data_metric2, data_metric3 = st.columns(3)

        data_metric1.metric(
            "Sample rows",
            f"{len(data):,}",
        )

        data_metric2.metric(
            "Dataset columns",
            len(data.columns),
        )

        if "exam_score" in data.columns:
            data_metric3.metric(
                "Average sample score",
                f"{data['exam_score'].mean():.1f}",
            )
        else:
            data_metric3.metric(
                "Average sample score",
                "N/A",
            )

        with st.expander("📋 View first 20 rows"):
            st.dataframe(
                data.head(20),
                width="stretch",
                hide_index=True,
            )

        with st.expander("📊 Dataset summary"):
            st.dataframe(
                data.describe().round(2),
                width="stretch",
            )

        # Study hours chart
        if {"study_hours", "exam_score"}.issubset(data.columns):
            chart_column1, chart_column2 = st.columns(2, gap="large")

            with chart_column1:
                fig, ax = plt.subplots(
                    figsize=(7, 4),
                    facecolor="#0D0F14",
                )

                ax.set_facecolor("#191B22")

                ax.scatter(
                    data["study_hours"],
                    data["exam_score"],
                    alpha=0.8,
                    color="#FFD400",
                    edgecolors="#EAB308",
                )

                ax.set_xlabel(
                    "Study hours per day",
                    color="#FFFFFF",
                )

                ax.set_ylabel(
                    "Exam score",
                    color="#FFFFFF",
                )

                ax.set_title(
                    "Study hours vs exam score",
                    color="#FFD400",
                    fontweight="bold",
                )

                ax.tick_params(colors="#FFFFFF")
                ax.grid(color="#444650", alpha=0.65)

                for spine in ax.spines.values():
                    spine.set_color("#555967")

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            # Attendance chart
            with chart_column2:
                if "attendance" in data.columns:
                    fig, ax = plt.subplots(
                        figsize=(7, 4),
                        facecolor="#0D0F14",
                    )

                    ax.set_facecolor("#191B22")

                    ax.scatter(
                        data["attendance"],
                        data["exam_score"],
                        alpha=0.8,
                        color="#FFD400",
                        edgecolors="#EAB308",
                    )

                    ax.set_xlabel(
                        "Attendance (%)",
                        color="#FFFFFF",
                    )

                    ax.set_ylabel(
                        "Exam score",
                        color="#FFFFFF",
                    )

                    ax.set_title(
                        "Attendance vs exam score",
                        color="#FFD400",
                        fontweight="bold",
                    )

                    ax.tick_params(colors="#FFFFFF")
                    ax.grid(color="#444650", alpha=0.65)

                    for spine in ax.spines.values():
                        spine.set_color("#555967")

                    fig.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

        st.download_button(
            "⬇️ Download sample dataset (CSV)",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="student_data_sample.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Could not load the sample dataset: {error}")

else:
    st.info(
        "Sample dataset not found. Check that "
        "`dataset/student_data.csv` exists."
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.divider()
st.subheader("🕘 Prediction history")

history = st.session_state.prediction_history

if history:
    history_dataframe = pd.DataFrame(history)

    st.dataframe(
        history_dataframe,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download prediction history (CSV)",
        data=history_dataframe.to_csv(index=False).encode("utf-8"),
        file_name="student_prediction_history.csv",
        mime="text/csv",
    )

    if st.button("🗑️ Clear session history"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No saved predictions in this session yet. "
        "Adjust the sidebar inputs and click Predict exam score."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Student Performance Predictor | Built with Python, "
    "Streamlit, Pandas and Scikit-learn. "
    "The bundled dataset is synthetic and predictions should "
    "not be used for real educational decisions."
)