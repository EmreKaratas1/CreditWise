# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from src.inference import predict_row, load_artifacts

st.set_page_config(page_title="CreditWise Segmentation", layout="wide")
st.title("CreditWise – Customer Segmentation (KMeans)")

# --- Paths (absolute, based on this file) ------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_DIR  = ROOT / "images" / "cluster_summaries"

# --- Load artifacts -----------------------------------------------------------
encoders, scaler, model, feature_cols, segnames = load_artifacts()

# --- Try dataset for dynamic options (fallback if missing) -------------------
try:
    df = pd.read_csv(DATA_DIR / "german_credit_data_clean.csv")
    job_opts = sorted(df["Job"].astype(int).unique().tolist())  # e.g., [0,1,2,3]
except Exception:
    job_opts = [0, 1, 2, 3]

# --- UI controls --------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.number_input("Age", min_value=18, max_value=75, value=45, step=1)
    duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=24, step=1)
    amount = st.number_input("Credit amount", min_value=250, max_value=20000, value=10000, step=50)

with col2:
    sex = st.selectbox("Sex", encoders["Sex"].classes_.tolist())
    job = st.selectbox("Job", job_opts, index=job_opts.index(2) if 2 in job_opts else 0)

col3, col4 = st.columns(2, gap="large")
with col3:
    housing = st.selectbox("Housing", encoders["Housing"].classes_.tolist())
    saving  = st.selectbox("Saving accounts", encoders["Saving accounts"].classes_.tolist())
with col4:
    checking = st.selectbox("Checking account", encoders["Checking account"].classes_.tolist())
    purpose  = st.selectbox("Purpose", encoders["Purpose"].classes_.tolist())

# --- Predict button (optional action) ----------------------------------------
if st.button("Predict segment"):
    row = {
        "Age": int(age),
        "Duration": int(duration),
        "Credit amount": int(amount),
        "Sex": sex,
        "Job": int(job),
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Purpose": purpose,
    }
    cid, cname = predict_row(row)
    st.success(f"Predicted: **{cid} — {cname}**")

# --- Always-on overview (table + visuals) ------------------------------------
st.markdown("---")
st.subheader("Cluster Overview")

# (1) Cluster profile table
try:
    prof = pd.read_csv(DATA_DIR / "cluster_profiles.csv")
    st.dataframe(prof, use_container_width=True)
except Exception:
    st.info(f"`{(DATA_DIR / 'cluster_profiles.csv').as_posix()}` bulunamadı.")

# Small helper to show images safely
def show_img(p: Path, caption: str):
    if p.exists():
        # Be compatible with both new and old Streamlit versions
        try:
            st.image(str(p), caption=caption, use_container_width=True)
        except TypeError:
            st.image(str(p), caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: `{p.as_posix()}`")


# (2) Visuals side-by-side
c1, c2 = st.columns(2, gap="large")
with c1:
    show_img(IMG_DIR / "housing_share_by_cluster.png", "Housing Share by Cluster")
with c2:
    show_img(IMG_DIR / "purpose_share_by_cluster.png", "Purpose (Top 6) Share by Cluster")

# (3) Notes / Markdown (report area)
st.markdown("### Notes")
st.markdown(
    """
- **Model:** K-Means (k=3)  
- **Preprocessing:** StandardScaler + Encoding  
- **Segment names:** Balanced, Younger/Short-Duration, High-Amount/Long-Duration  
- **Observation:** Purpose distributions are pronounced across segments; “car/business” → tendency toward Segment 2.
"""
)
