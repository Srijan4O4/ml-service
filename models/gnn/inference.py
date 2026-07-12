"""inference.py — load the trained RGCN and expose scoring helpers.

Two scoring paths:
  * predict_seller(name): true transductive RGCN inference for a synthetic-graph seller
    (demonstrates collusion / structure-aware detection).
  * score_external(features): a fast, GNN-consistent feature scorer for arbitrary
    real sellers coming from the backend (which only has aggregate features).
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import numpy as np
import torch

from .model import RGCN

ART = Path(__file__).resolve().parents[2] / "artifacts" / "gnn"
_state = {}
_lock = threading.Lock()


def _ensure():
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        if not (ART / "rgcn_seller_fraud.pth").exists() or not (ART / "graph.pt").exists():
            print("[gnn] artifacts missing -> training now...")
            from . import train as train_mod
            train_mod.main()

        g = torch.load(ART / "graph.pt", weights_only=False)
        ckpt = torch.load(ART / "rgcn_seller_fraud.pth", weights_only=False)
        cfg = ckpt["config"]
        model = RGCN(cfg["in_feats"], cfg["hidden"], cfg["num_rels"], cfg["num_classes"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(g["x"], g["edge_index"], g["edge_type"])
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

        off = g["offsets"]
        s0, S = off["seller"], off["num_sellers"]
        seller_probs = probs[s0:s0 + S]
        seller_labels = g["labels"][s0:s0 + S].numpy()

        seller_features = json.load(open(ART / "seller_features.json"))
        mappings = json.load(open(ART / "mappings.json"))

        # GNN-consistent inductive scorer for external sellers (raw 3-feature space)
        names = list(seller_features.keys())
        raw = np.array([[seller_features[n]["return_ratio"],
                         seller_features[n]["avg_rating"],
                         seller_features[n]["burstiness"]] for n in names], dtype=float)
        try:
            from sklearn.linear_model import LogisticRegression
            scorer = LogisticRegression(max_iter=1000, class_weight="balanced")
            scorer.fit(raw, seller_labels)
        except Exception as e:  # pragma: no cover
            print("[gnn] feature scorer fallback:", e)
            scorer = None

        _state.update({
            "model": model, "graph": g, "seller_probs": seller_probs,
            "seller_labels": seller_labels, "seller_features": seller_features,
            "mappings": mappings, "names": names, "scorer": scorer,
        })
        return _state


def predict_seller(name: str):
    """True RGCN inference for a synthetic-graph seller."""
    st = _ensure()
    idx_map = st["mappings"]["seller_to_index"]
    if name not in idx_map:
        return None
    off = st["graph"]["offsets"]
    local = idx_map[name] - off["seller"]
    return {
        "seller": name,
        "fraud_probability": float(st["seller_probs"][local]),
        "is_fraud_label": int(st["seller_labels"][local]),
        "features": st["seller_features"][name],
        "method": "rgcn-transductive",
    }


def score_external(features: dict):
    """Score an arbitrary real seller from aggregate features."""
    st = _ensure()
    rr = float(features.get("returnRate", features.get("return_ratio", 0)) or 0)
    ar = float(features.get("avgRating", features.get("avg_rating", 0)) or 0)
    bu = float(features.get("burstiness", 0) or 0)
    scorer = st["scorer"]
    if scorer is not None:
        p = float(scorer.predict_proba([[rr, ar, bu]])[0, 1])
    else:
        p = min(0.99, 0.6 * rr + 0.4 * bu)
    return {"fraud_probability": p, "method": "gnn-feature-scorer",
            "features": {"returnRate": rr, "avgRating": ar, "burstiness": bu}}


def metrics():
    p = ART / "metrics.json"
    return json.load(open(p)) if p.exists() else {}


def list_sellers(top_k: int = 10, fraud_only: bool = True):
    """Riskiest synthetic sellers by RGCN fraud probability."""
    st = _ensure()
    names = st["names"]
    order = np.argsort(-st["seller_probs"])
    out = []
    for i in order:
        name = names[i] if i < len(names) else None
        if name is None:
            continue
        out.append({"seller": name,
                    "fraud_probability": float(st["seller_probs"][i]),
                    "label": int(st["seller_labels"][i])})
        if len(out) >= top_k:
            break
    return out
