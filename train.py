# train.py
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
from text_utils import normalize_text

# 1️⃣ Set project directory (portable)
BASE_DIR = Path(__file__).resolve().parent
print("Using project directory:", BASE_DIR)

# 2️⃣ Load datasets
fake_csv_path = BASE_DIR / "Fake.csv"
real_csv_path = BASE_DIR / "True.csv"

fake_df = pd.read_csv(fake_csv_path)
real_df = pd.read_csv(real_csv_path)

# 3️⃣ Label datasets
fake_df["label"] = 0   # Fake
real_df["label"] = 1   # Real

# 4️⃣ Combine datasets
df = pd.concat([fake_df, real_df], axis=0).reset_index(drop=True)

# Optional: combine title + text
if "title" in df.columns:
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

# Drop empty/duplicate texts
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

# 5️⃣ Features and labels
X = df["text"].astype(str)
y = df["label"].astype(int)

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Define feature pipelines
word_hash = Pipeline([
    ("hash", HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        preprocessor=normalize_text,
    )),
    ("tfidf", TfidfTransformer(sublinear_tf=True, use_idf=True)),
])

char_hash = Pipeline([
    ("hash", HashingVectorizer(
        n_features=2**19,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        preprocessor=normalize_text,
    )),
    ("tfidf", TfidfTransformer(sublinear_tf=True, use_idf=True)),
])

features = FeatureUnion([
    ("word", word_hash),
    ("char", char_hash),
], transformer_weights={
    "word": 0.85,
    "char": 0.15
})

# 8️⃣ Classifier
clf = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    max_iter=120,
    tol=1e-4,
    alpha=5e-6,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    learning_rate="optimal",
)

# Full pipeline
pipe = Pipeline([
    ("features", features),
    ("clf", clf),
])

# Train
pipe.fit(X_train, y_train)

# 9️⃣ Evaluate
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report_text = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred).tolist()

# ROC-based threshold
try:
    y_scores = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    youden_j = tpr - fpr
    best_idx = youden_j.argmax()
    best_threshold = float(thresholds[best_idx])
except Exception:
    best_threshold = 0.5

print(f"Accuracy: {acc:.4f}")
print("Classification report:\n" + report_text)
print(f"Suggested decision threshold (Real=1): {best_threshold:.3f}")

# 🔥 Save only ONE artifact (pipeline with vectorizer + model)
model_pipeline_path = BASE_DIR / "model_pipeline.pkl"
joblib.dump(pipe, model_pipeline_path)

# Save metrics
reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(exist_ok=True)
with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": acc,
        "report": report_dict,
        "confusion_matrix": cm,
        "recommended_threshold": best_threshold
    }, f, indent=2)

print("✅ Training complete. Model pipeline saved as model_pipeline.pkl")
