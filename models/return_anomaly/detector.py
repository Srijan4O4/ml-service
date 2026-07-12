"""detector.py — Return-Rate Anomaly Monitoring (built from scratch).

Detects sellers/products whose return behaviour deviates from normal patterns
(e.g. a sudden weekly spike to 70% returns). Combines:
  * an IsolationForest trained on synthetic "normal" return-rate windows, and
  * robust statistical spike rules (z-score / category threshold).

API:
  train()                    -> fits & saves artifacts/return_anomaly/iforest.joblib
  detect(series, category)   -> {anomaly, score, threshold, reason, ...}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[2] / "artifacts" / "return_anomaly"
ART.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART / "iforest.joblib"
META_PATH = ART / "meta.json"

WINDOW = 8
RNG = np.random.default_rng(7)


def _features(rates: np.ndarray) -> np.ndarray:
    """Feature vector summarising a return-rate window."""
    rates = np.asarray(rates, dtype=float)
    if len(rates) == 0:
        return np.zeros(6)
    last = rates[-1]
    mean = rates.mean()
    std = rates.std() + 1e-6
    mx = rates.max()
    slope = np.polyfit(np.arange(len(rates)), rates, 1)[0] if len(rates) > 1 else 0.0
    z = (last - mean) / std
    return np.array([mean, std, mx, last, slope, z])


def _make_normal_window():
    base = RNG.uniform(0.03, 0.18)
    season = 0.02 * np.sin(np.linspace(0, RNG.uniform(1, 3) * np.pi, WINDOW))
    noise = RNG.normal(0, 0.02, WINDOW)
    return np.clip(base + season + noise, 0, 0.95)


def train(n_normal: int = 4000):
    from sklearn.ensemble import IsolationForest
    import joblib

    X = np.array([_features(_make_normal_window()) for _ in range(n_normal)])
    iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iforest.fit(X)

    # calibrate a decision threshold on held-out normal + injected anomalies
    normals = np.array([_features(_make_normal_window()) for _ in range(500)])
    anomalies = []
    for _ in range(500):
        w = _make_normal_window()
        w[-1] = RNG.uniform(0.45, 0.85)  # inject a spike
        anomalies.append(_features(w))
    anomalies = np.array(anomalies)
    s_norm = -iforest.score_samples(normals)
    s_anom = -iforest.score_samples(anomalies)
    thr = float(np.quantile(s_norm, 0.95))
    detect_rate = float((s_anom > thr).mean())

    joblib.dump(iforest, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "window": WINDOW, "threshold": thr,
        "anomaly_detect_rate": round(detect_rate, 3),
        "normal_fpr": round(float((s_norm > thr).mean()), 3),
    }, indent=2))
    print(f"[return-anomaly] trained. threshold={thr:.4f} detect_rate={detect_rate:.3f}")
    return {"threshold": thr, "detect_rate": detect_rate}


_cache = {}


def _load():
    if _cache:
        return _cache
    import joblib
    if not MODEL_PATH.exists():
        train()
    _cache["model"] = joblib.load(MODEL_PATH)
    _cache["meta"] = json.loads(META_PATH.read_text())
    return _cache


def detect(series, category: str = "General"):
    """series: list of numbers OR list of {returnRate:..} dicts."""
    rates = []
    for s in (series or []):
        if isinstance(s, dict):
            rates.append(float(s.get("returnRate", s.get("rate", 0)) or 0))
        else:
            rates.append(float(s))
    rates = np.array(rates[-WINDOW:]) if rates else np.array([])

    if len(rates) == 0:
        return {"anomaly": False, "score": 0.0, "reason": "no data"}

    st = _load()
    feats = _features(rates).reshape(1, -1)
    raw = float(-st["model"].score_samples(feats)[0])
    thr = st["meta"]["threshold"]
    model_flag = raw > thr

    # statistical spike rules (robust)
    last = float(rates[-1])
    mean = float(rates[:-1].mean()) if len(rates) > 1 else float(rates.mean())
    std = float(rates[:-1].std()) if len(rates) > 1 else 0.0
    z = (last - mean) / (std + 1e-6)
    rule_flag = last >= 0.4 or (z >= 3 and last >= 0.25)

    anomaly = bool(model_flag or rule_flag)
    reasons = []
    if last >= 0.4:
        reasons.append(f"return rate {last:.0%} exceeds 40% norm")
    if z >= 3:
        reasons.append(f"spike {z:.1f} SD above recent baseline")
    if model_flag and not reasons:
        reasons.append("isolation-forest outlier")

    return {
        "anomaly": anomaly,
        "score": round(raw, 4),
        "threshold": round(thr, 4),
        "lastReturnRate": round(last, 4),
        "baselineMean": round(mean, 4),
        "zScore": round(z, 2),
        "reason": "; ".join(reasons) if reasons else "within normal range",
        "method": "isolation-forest + statistical-rules",
    }


def metrics():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


if __name__ == "__main__":
    train()
    print("normal :", detect([0.05, 0.07, 0.06, 0.08, 0.05, 0.07, 0.06, 0.09]))
    print("spike  :", detect([0.05, 0.07, 0.06, 0.08, 0.05, 0.07, 0.06, 0.72]))
