"""app.py — BEACON ML service (FastAPI).

Exposes all six Trust & Safety models behind one HTTP API that the Node backend calls.
Each model is imported lazily and wrapped so one failing model never breaks the service.

Run:  uvicorn app:app --port 8000   (from the ml-service/ directory)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="BEACON ML Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------- schemas -----------------------------
class ReviewReq(BaseModel):
    text: str = ""
    rating: Any = None


class SemanticReq(BaseModel):
    name: str = ""
    description: str = ""


class ImageReq(BaseModel):
    imagePaths: list[str] = []
    productName: str = ""


class SellerSuspicionReq(BaseModel):
    returnRate: Any = ""
    avgRating: Any = ""
    recentReviews: list[str] = []


class GraphReq(BaseModel):
    sellerName: str = ""
    features: dict = {}


class ReturnsReq(BaseModel):
    series: list = []
    category: str = "General"


def _safe(fn, fallback):
    try:
        return fn()
    except Exception as e:  # noqa
        return {**fallback, "error": str(e), "_degraded": True}


# ----------------------------- routes -----------------------------
@app.get("/health")
def health():
    from common import gemini
    return {
        "status": "ok",
        "service": "beacon-ml",
        "gemini": gemini.available(),
        "models": ["review", "semantic", "image", "seller", "graph", "returns"],
    }


@app.post("/ml/review/authenticity")
def review_authenticity(req: ReviewReq):
    from models.review_authenticity.predict import predict
    return _safe(lambda: predict(req.text, req.rating),
                 {"isAI": False, "aiScore": 0.0, "authenticity": "unknown"})


@app.post("/ml/semantic/validate")
def semantic_validate(req: SemanticReq):
    from models.seller_suspicion.classify import validate_semantic
    return _safe(lambda: validate_semantic(req.name, req.description),
                 {"riskScore": 0.0, "flags": [], "deceptive": False})


@app.post("/ml/image/verify")
def image_verify(req: ImageReq):
    from models.image_logo.verify import verify
    return _safe(lambda: verify(req.imagePaths, req.productName),
                 {"fakeProbability": 0.1, "label": "unverified", "flags": []})


@app.post("/ml/seller/suspicion")
def seller_suspicion(req: SellerSuspicionReq):
    from models.seller_suspicion.classify import classify_seller
    return _safe(lambda: classify_seller(req.returnRate, req.avgRating, req.recentReviews),
                 {"classification": "not suspicious", "confidence": 50})


@app.post("/ml/graph/score")
def graph_score(req: GraphReq):
    from models.gnn import inference as gnn
    def run():
        # exact synthetic seller -> transductive RGCN; else inductive feature scorer
        res = gnn.predict_seller(req.sellerName) if req.sellerName else None
        if res:
            return res
        return gnn.score_external(req.features or {})
    return _safe(run, {"fraud_probability": 0.0})


@app.get("/ml/graph/seller/{name}")
def graph_seller(name: str):
    from models.gnn import inference as gnn
    return _safe(lambda: gnn.predict_seller(name) or {"error": "unknown seller"},
                 {"fraud_probability": 0.0})


@app.get("/ml/graph/top-fraud")
def graph_top(top_k: int = 10):
    from models.gnn import inference as gnn
    return _safe(lambda: {"sellers": gnn.list_sellers(top_k)}, {"sellers": []})


@app.post("/ml/returns/anomaly")
def returns_anomaly(req: ReturnsReq):
    from models.return_anomaly.detector import detect
    return _safe(lambda: detect(req.series, req.category),
                 {"anomaly": False, "score": 0.0})


@app.get("/ml/metrics")
def ml_metrics():
    out = {}
    try:
        from models.gnn import inference as gnn
        out["gnn"] = gnn.metrics()
    except Exception as e:
        out["gnn"] = {"error": str(e)}
    try:
        from models.return_anomaly.detector import metrics as ra_metrics
        out["return_anomaly"] = ra_metrics()
    except Exception as e:
        out["return_anomaly"] = {"error": str(e)}
    try:
        import json
        p = Path(__file__).resolve().parent / "artifacts" / "review_authenticity" / "tfidf_metrics.json"
        out["review"] = json.loads(p.read_text()) if p.exists() else {}
    except Exception as e:
        out["review"] = {"error": str(e)}
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ML_PORT", "8000")))
