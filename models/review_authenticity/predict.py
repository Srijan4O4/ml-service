"""predict.py — Review Authenticity inference.

Load order (first available wins):
  1. artifacts/review_authenticity/distilbert/   (optional retrained DistilBERT)
  2. artifacts/review_authenticity/tfidf_lr.joblib (TF-IDF + LogisticRegression, ~96% acc)
  3. lexical heuristic fallback

The DistilBERT checkpoint that originally shipped with the project had collapsed to
majority-class predictions, so it is intentionally NOT used.
"""
from __future__ import annotations
import re
import threading
from pathlib import Path

ART = Path(__file__).resolve().parents[2] / "artifacts" / "review_authenticity"
BERT_DIR = ART / "distilbert"
TFIDF_PATH = ART / "tfidf_lr.joblib"

_state = {}
_lock = threading.Lock()

_GENERIC = re.compile(
    r"(great product|highly recommend|exactly as described|good quality|love it|"
    r"amazing product|works perfectly|value for money|must buy|best purchase)", re.I)


def _heuristic(text, rating=None):
    t = (text or "").strip()
    tl = t.lower()
    score = 0.12
    if _GENERIC.search(tl):
        score += 0.45
    if len(t) < 60:
        score += 0.2
    words = re.findall(r"[a-z']+", tl)
    if words and len(set(words)) / len(words) < 0.6:
        score += 0.15
    score = max(0.0, min(0.96, score))
    return {"isAI": score >= 0.5, "aiScore": round(score, 4),
            "authenticity": "AI-generated" if score >= 0.5 else "Human-written",
            "method": "heuristic"}


def _build_text(text, rating):
    return f"{text or ''} Rating: {rating}" if rating is not None else (text or "")


def _load():
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        # 1) retrained DistilBERT (optional)
        if BERT_DIR.exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
                model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
                model.eval()
                _state.update({"kind": "distilbert", "tok": tok, "model": model, "torch": torch})
                print(f"[review] using retrained DistilBERT ({BERT_DIR})")
                return _state
            except Exception as e:
                print(f"[review] DistilBERT load failed: {e}")
        # 2) TF-IDF + LogReg
        if TFIDF_PATH.exists():
            try:
                import joblib
                _state.update({"kind": "tfidf", "pipe": joblib.load(TFIDF_PATH)})
                print(f"[review] using TF-IDF+LogReg ({TFIDF_PATH})")
                return _state
            except Exception as e:
                print(f"[review] TF-IDF load failed: {e}")
        # 3) heuristic
        _state["kind"] = "heuristic"
        print("[review] no trained model found; using heuristic")
        return _state


def predict(text: str, rating=None):
    st = _load()
    kind = st.get("kind")

    if kind == "distilbert":
        torch = st["torch"]
        enc = st["tok"](_build_text(text, rating), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probs = torch.softmax(st["model"](**enc).logits, dim=1)[0].tolist()
        ai = float(probs[1]) if len(probs) > 1 else float(probs[-1])
        return {"isAI": ai >= 0.5, "aiScore": round(ai, 4),
                "authenticity": "AI-generated" if ai >= 0.5 else "Human-written", "method": "distilbert"}

    if kind == "tfidf":
        pipe = st["pipe"]
        x = _build_text(text, rating)
        try:
            ai = float(pipe.predict_proba([x])[0][1])
        except Exception:
            ai = float(pipe.predict([x])[0])
        # ensemble: boost obvious boilerplate the dataset model under-weights
        ai = max(ai, _heuristic(text, rating)["aiScore"] if _GENERIC.search((text or "").lower()) else ai)
        return {"isAI": ai >= 0.5, "aiScore": round(ai, 4),
                "authenticity": "AI-generated" if ai >= 0.5 else "Human-written", "method": "tfidf-lr+rules"}

    return _heuristic(text, rating)


if __name__ == "__main__":
    print(predict("Amazing product, highly recommend! Exactly as described, works perfectly. Love it!", 5))
    print(predict("Arrived with a small dent but the speaker still works fine after a week of daily use.", 4))
    print(predict("Their Congress no throughout successful owner.", 3))
