from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_split(name):
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    # --- NEW: drop or patch bad rows ------------------------
    df = df.dropna(subset=["text"]).copy()          # remove rows whose text is None
    df["text"] = df["text"].astype(str).str.strip() # cast to str, strip whitespace
    return df

train_df, val_df, test_df = map(load_split, ["train", "valid", "test"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
         lowercase=True, stop_words="english", ngram_range=(1,2),
         max_features=30_000)),
    ("clf", LogisticRegression(max_iter=2_000, n_jobs=-1))
])

pipe.fit(train_df.text, train_df.label)

# -------- metrics on valid ----------
y_pred = pipe.predict(val_df.text)
report = classification_report(val_df.label, y_pred, output_dict=True)
print(json.dumps(report, indent=2))

# -------- persist artefacts ----------
MODELS = Path("artefacts"); MODELS.mkdir(exist_ok=True)
joblib.dump(pipe, MODELS / "baseline_logreg.joblib")
json.dump(report, open(MODELS / "baseline_metrics.json", "w"))