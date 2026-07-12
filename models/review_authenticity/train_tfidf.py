"""train_tfidf.py — fast, reliable Review Authenticity classifier (TF-IDF + LogisticRegression).

The shipped DistilBERT checkpoint had collapsed to majority-class (49% acc), so we train
a dependable CPU model on the balanced merged_dataset.csv (66k human/AI reviews).
A DistilBERT retrain is available separately (train_bert.py); predict.py prefers whichever
trained artifact exists.

Run:  python -m models.review_authenticity.train_tfidf
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parents[2] / "artifacts" / "review_authenticity"
ART.mkdir(parents=True, exist_ok=True)
DATA = os.environ.get("REVIEW_DATA_CSV", r"D:\Amazon HackOn S5\Review Analysis Model\merged_dataset.csv")


def build_text(df):
    rating = df["rating"].astype(str) if "rating" in df.columns else ""
    review = df["review"].astype(str) if "review" in df.columns else df["review_text"].astype(str)
    return review + " Rating: " + rating if "rating" in df.columns else review


def main():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    print(f"[review-tfidf] loading {DATA}")
    df = pd.read_csv(DATA).dropna(subset=["review", "isAI"])
    df = df[df["isAI"].isin([0, 1])]
    X = build_text(df)
    y = df["isAI"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(C=4.0, max_iter=400, n_jobs=-1)),
    ])
    print(f"[review-tfidf] training on {len(Xtr)} samples...")
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    print(f"[review-tfidf] TEST accuracy={acc:.4f} f1={f1:.4f}")
    print(classification_report(yte, pred, target_names=["Human-written", "AI-generated"]))

    joblib.dump(pipe, ART / "tfidf_lr.joblib")
    (ART / "tfidf_metrics.json").write_text(
        __import__("json").dumps({"accuracy": round(acc, 4), "f1": round(f1, 4), "n_train": len(Xtr)}, indent=2))
    print(f"[review-tfidf] saved -> {ART / 'tfidf_lr.joblib'}")
    return acc


if __name__ == "__main__":
    main()
