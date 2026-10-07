import json
import pickle

import joblib
import re

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- File paths (keep all four files in the same folder as app.py) ----------
MODEL_PATH = "financial_lstm.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
CONFIG_PATH = "config.json"

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈")


# ---------- Load artifacts once (cached across reruns) ----------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    label_encoder = joblib.load(ENCODER_PATH)  # saved with joblib, not pickle
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, tokenizer, label_encoder, config


model, tokenizer, label_encoder, config = load_artifacts()
MAX_LEN = config["max_len"]
STOP_WORDS = set(config["stop_words"])  # negation words (no/not/nor) are already excluded
CLASSES = list(label_encoder.classes_)


# ---------- Preprocessing: must match what you did during training ----------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # adjust if your training cleaning differed
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


def predict(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    # padding/truncating must match training (Keras default is 'pre')
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    probs = model.predict(padded, verbose=0)[0]
    return cleaned, probs


# ---------- UI ----------
st.title("📈 Financial Sentiment Analyzer")
st.caption("LSTM model: classifies financial text as negative, neutral or positive.")

EXAMPLES = {
    "Positive": "The company's quarterly profit rose 20% beating analyst expectations.",
    "Negative": "Shares plunged after the firm reported a sharp drop in revenue.",
    "Neutral": "The board will meet on Monday to discuss the annual report.",
}

with st.sidebar:
    st.header("Try an example")
    choice = st.selectbox("Example sentence", ["(none)"] + list(EXAMPLES))
    st.markdown(f"**Max sequence length:** {MAX_LEN}")

default_text = EXAMPLES.get(choice, "")
user_text = st.text_area("Enter a financial sentence or headline:", value=default_text, height=120)

if st.button("Analyze", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned, probs = predict(user_text)
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(label, "")

        st.subheader(f"{icon} Prediction: {label.capitalize()}")
        st.metric("Confidence", f"{probs[pred_idx] * 100:.1f}%")

        chart_df = pd.DataFrame({"Class": CLASSES, "Probability": probs}).set_index("Class")
        st.bar_chart(chart_df)

        with st.expander("Preprocessed text"):
            st.write(cleaned if cleaned else "(empty after cleaning)")
            