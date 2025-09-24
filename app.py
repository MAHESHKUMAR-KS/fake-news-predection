import streamlit as st
import joblib
from pathlib import Path
import re
import unicodedata

# Define normalize_text at module scope so joblib can resolve it when unpickling
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s,\.!?\-\'\"]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Safe load utilities
@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        pipe = joblib.load("model_pipeline.pkl")
        return pipe, None
    except Exception as e:
        return None, str(e)

pipe, load_err = load_pipeline()

st.title("Fake News Detection 🚨")
st.write("Paste a news article or headline below and click Predict:")

if load_err:
    st.error(
        "Failed to load pipeline. Make sure 'model_pipeline.pkl' is present.\n\n" + str(load_err)
    )
    st.stop()

text = st.text_area("News text", height=200)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to predict!")
    elif len(text.split()) < 5:
        st.warning("⚠️ Text is too short. Please enter at least 5 words for reliable prediction.")
    else:
        # Final decision strictly from pipeline.predict()
        pred = int(pipe.predict([text])[0])  # 1 -> Real, 0 -> Fake

        # Probabilities (if available) for display only
        p_real = None
        p_fake = None
        try:
            proba = pipe.predict_proba([text])[0]
            p_fake = float(proba[0])
            p_real = float(proba[1])
        except Exception:
            pass

        # Display result
        if pred == 1:
            st.success("Prediction: Real News ✅")
        else:
            st.error("Prediction: Fake News ❌")

        with st.expander("Details"):
            details = {"predicted_label": int(pred)}
            if p_real is not None:
                details.update({"p_real": round(p_real, 4), "p_fake": round(p_fake, 4)})
            st.write(details)
