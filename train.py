# train_kaggle_compatible.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import joblib
import json
import re
import unicodedata
import os

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


input_dir = "/kaggle/input/"
dataset_folder = None

for folder in os.listdir(input_dir):
    if set(["Fake.csv", "True.csv"]).issubset(os.listdir(os.path.join(input_dir, folder))):
        dataset_folder = os.path.join(input_dir, folder)
        break

if dataset_folder is None:
    raise FileNotFoundError("Could not find Fake.csv and True.csv in /kaggle/input/")

FAKE_CSV = Path(dataset_folder) / "Fake.csv"
REAL_CSV = Path(dataset_folder) / "True.csv"

MODEL_PIPELINE_PATH = Path("/kaggle/working/model_pipeline.pkl")
REPORTS_DIR = Path("/kaggle/working/reports")
REPORTS_DIR.mkdir(exist_ok=True)

print(f"Using dataset folder: {dataset_folder}")

# ----------------------------
# Load datasets
# ----------------------------
fake_df = pd.read_csv(FAKE_CSV)
real_df = pd.read_csv(REAL_CSV)

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)

# Combine title + text if available
if "title" in df.columns:
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

X = df["text"].astype(str)
y = df["label"].astype(int)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Feature pipelines
# ----------------------------
word_hash = Pipeline([
    ("hash", HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        analyzer="word",
        ngram_range=(1,2),
        stop_words="english",
        lowercase=True,
        preprocessor=normalize_text
    )),
    ("tfidf", TfidfTransformer(sublinear_tf=True, use_idf=True)),
])

char_hash = Pipeline([
    ("hash", HashingVectorizer(
        n_features=2**17,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3,4),
        lowercase=True,
        preprocessor=normalize_text
    )),
    ("tfidf", TfidfTransformer(sublinear_tf=True, use_idf=True)),
])

features = FeatureUnion([
    ("word", word_hash),
    ("char", char_hash)
], transformer_weights={"word": 0.85, "char": 0.15})

# ----------------------------
# Classifier (compatible)
# ----------------------------
clf = SGDClassifier(
    loss="log",               # use 'log' for backward compatibility
    class_weight="balanced",
    max_iter=50,
    tol=1e-4,
    alpha=5e-6,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    learning_rate="optimal"
)

pipe = Pipeline([
    ("features", features),
    ("clf", clf)
])

# ----------------------------
# Train
# ----------------------------
print("Training model ...")
pipe.fit(X_train, y_train)
print("Training complete ✅")

# ----------------------------
# Evaluate
# ----------------------------
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

try:
    y_scores = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    youden_j = tpr - fpr
    best_idx = youden_j.argmax()
    best_threshold = float(thresholds[best_idx])
except Exception:
    best_threshold = 0.5

print(f"Accuracy: {acc:.4f}")
print(f"Recommended threshold: {best_threshold:.3f}")

# ----------------------------
# Save pipeline & metrics
# ----------------------------
joblib.dump(pipe, MODEL_PIPELINE_PATH)

with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report_dict,
        "recommended_threshold": best_threshold
    }, f, indent=2)

print(f"Model pipeline saved at {MODEL_PIPELINE_PATH}")
