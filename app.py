import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Health ML System", page_icon="🩺", layout="wide")
st.title("🩺 Multiple Disease Prediction System")
st.warning("Educational project only. This is NOT a medical diagnosis. "
           "Please consult a qualified doctor.")


@st.cache_resource
def load_bundle(tag):
    return joblib.load(f"models/{tag}_model.joblib")


def render(tag, title, positive_msg, negative_msg):
    bundle = load_bundle(tag)
    pipe, features, meta = bundle["pipeline"], bundle["features"], bundle["meta"]

    st.subheader(title)
    st.caption(f"Model: {bundle['model_name']} | Test ROC-AUC: {bundle['test_auc']:.2f}")

    cols = st.columns(3)
    values = {}
    for i, f in enumerate(features):
        m = meta[f]
        with cols[i % 3]:
            if m["is_int"]:
                values[f] = st.number_input(
                    f, min_value=int(m["min"]), max_value=int(m["max"]),
                    value=int(round(m["median"])), step=1, key=f"{tag}_{f}")
            else:
                fmt = "%.6f" if m["max"] < 1 else "%.2f"
                step = float((m["max"] - m["min"]) / 100) or 0.01
                values[f] = st.number_input(
                    f, min_value=float(m["min"]), max_value=float(m["max"]),
                    value=float(m["median"]), step=step, format=fmt,
                    key=f"{tag}_{f}")

    if st.button(f"Check {title}", key=f"btn_{tag}"):
        row = pd.DataFrame([values], columns=features)
        pred = pipe.predict(row)[0]
        proba = pipe.predict_proba(row)[0][1]
        if pred == 1:
            st.error(positive_msg)
        else:
            st.success(negative_msg)
        st.progress(float(proba))
        st.write(f"Model probability of positive class: **{proba:.1%}**")


tab1, tab2, tab3 = st.tabs(["Diabetes", "Heart Disease", "Parkinson's"])
with tab1:
    render("diabetes", "Diabetes Prediction",
           "The model predicts: likely diabetic.",
           "The model predicts: likely not diabetic.")
with tab2:
    render("heart", "Heart Disease Prediction",
           "The model predicts: likely heart disease.",
           "The model predicts: likely no heart disease.")
with tab3:
    render("parkinsons", "Parkinson's Prediction",
           "The model predicts: likely Parkinson's.",
           "The model predicts: likely healthy.")