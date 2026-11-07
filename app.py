"""Streamlit interface for the MartechMatch lead-scoring pipeline."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "martech_pipeline.pkl"
INDUSTRIES = ["Tech", "Healthcare", "Finance", "Retail"]
COMPANY_SIZES = ["Startup", "Mid-Market", "Enterprise"]
INPUT_COLUMNS = [
    "Time_on_Website",
    "Email_Open_Rate",
    "Webinar_Attendance",
    "Industry",
    "Company_Size",
]

st.set_page_config(
    page_title="MartechMatch | Lead Scoring",
    page_icon="🎯",
    layout="centered",
)


@st.cache_resource
def load_pipeline(model_path: str):
    """Load and cache the fitted scikit-learn pipeline."""
    return joblib.load(model_path)


st.title("🎯 MartechMatch")
st.subheader("Predictive Lead Scoring")
st.caption(
    "Estimate conversion likelihood from engagement and firmographic signals. "
    "Use the score to support, not replace, sales judgment."
)

if not MODEL_PATH.is_file():
    st.error(
        "Trained model not found. From the project directory, run "
        "`python train_model.py` first."
    )
    st.stop()

try:
    pipeline = load_pipeline(str(MODEL_PATH))
except (OSError, EOFError, ValueError, AttributeError) as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

with st.sidebar:
    st.header("Lead profile")
    st.caption("Enter the lead's recent engagement and company details.")
    time_on_website = st.slider(
        "Time on website (seconds)",
        min_value=0,
        max_value=1_800,
        value=240,
        step=10,
        help="Total time spent on your website during the measured period.",
    )
    email_open_rate = st.slider(
        "Email open rate",
        min_value=0.0,
        max_value=1.0,
        value=0.42,
        step=0.01,
        format="%.2f",
        help="Share of delivered marketing emails opened, from 0 to 1.",
    )
    webinar_attendance = st.radio(
        "Attended a webinar?",
        options=[0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No",
        horizontal=True,
    )
    industry = st.selectbox("Industry", options=INDUSTRIES, index=0)
    company_size = st.selectbox("Company size", options=COMPANY_SIZES, index=1)

st.markdown("### Lead details")
summary_left, summary_right = st.columns(2)
summary_left.metric("Website time", f"{time_on_website:,} sec")
summary_right.metric("Email open rate", f"{email_open_rate:.0%}")

if st.button("Predict Conversion", type="primary", use_container_width=True):
    lead = pd.DataFrame(
        [
            {
                "Time_on_Website": float(time_on_website),
                "Email_Open_Rate": float(email_open_rate),
                "Webinar_Attendance": int(webinar_attendance),
                "Industry": industry,
                "Company_Size": company_size,
            }
        ],
        columns=INPUT_COLUMNS,
    )

    try:
        probabilities = pipeline.predict_proba(lead)[0]
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(1)
        conversion_probability = float(probabilities[positive_index])
        predicted_class = int(pipeline.predict(lead)[0])
    except (AttributeError, IndexError, KeyError, ValueError) as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.markdown("### Prediction")
    st.metric("Conversion probability", f"{conversion_probability:.2%}")
    st.progress(conversion_probability)

    if predicted_class == 1:
        st.success("**High-Value Lead: Route to Sales**")
    else:
        st.info("**Low-Value Lead: Keep in Nurture Campaign**")

    st.caption(
        "The probability is the model's estimate. Validate the scoring threshold "
        "against your team's capacity and observed conversion costs."
    )
