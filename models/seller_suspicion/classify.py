"""classify.py — Seller Suspicion + Semantic Description Validation (LLM).

Uses Gemini to (a) classify a seller as suspicious/not from aggregate signals, and
(b) validate listing text for deceptive/unauthorized claims. Both degrade gracefully
to deterministic heuristics when no API key / network is available.
"""
from __future__ import annotations
import json
import re

from common import gemini

_CLAIM = re.compile(r"(official|licensed|authorized|genuine|100% original|replica|authentic)", re.I)
_HYPE = re.compile(r"(best|cheapest|guaranteed|miracle|#1|unbeatable|lowest price)", re.I)


# ---------------- seller suspicion ----------------
def classify_seller(return_rate, average_rating, recent_reviews=None, temperature=0.0):
    reviews = recent_reviews or []
    if gemini.available():
        bullets = "\n  - " + "\n  - ".join(reviews) if reviews else " (none)"
        prompt = (
            "You are a marketplace trust-and-safety analyst. Classify the seller as "
            "exactly 'suspicious' or 'not suspicious', then give a one-line justification "
            "and a confidence score 0-100.\n"
            "Respond strictly as:\nClassification: <label>\nJustification: <text>\nConfidence: <score>\n\n"
            f"Seller data:\n- Return rate: {return_rate}\n- Average rating: {average_rating}\n"
            f"- Recent reviews:{bullets}\n"
        )
        res = gemini.generate(prompt, temperature=temperature)
        if res["ok"] and res["text"]:
            parsed = _parse_labeled(res["text"])
            if parsed.get("classification"):
                parsed["method"] = "gemini"
                return parsed
    return {**_seller_heuristic(return_rate, average_rating, reviews), "method": "heuristic"}


def _seller_heuristic(return_rate, average_rating, reviews):
    rr = _to_float(return_rate)
    ar = _to_float(average_rating)
    generic = sum(bool(re.search(r"(highly recommend|amazing|love it|works perfectly)", r, re.I)) for r in reviews)
    suspicious = rr > 0.4 or (generic >= max(2, len(reviews) // 2) and ar >= 4.5)
    conf = 70 if suspicious else 60
    reason = []
    if rr > 0.4:
        reason.append(f"high return rate ({return_rate})")
    if generic:
        reason.append("repetitive/boilerplate reviews")
    return {
        "classification": "suspicious" if suspicious else "not suspicious",
        "confidence": conf,
        "justification": ", ".join(reason) or "signals within normal range",
    }


# ---------------- semantic / description validation ----------------
def validate_semantic(name, description, temperature=0.0):
    text = f"{name or ''}\n{description or ''}".strip()
    if gemini.available():
        prompt = (
            "Assess this product listing for DECEPTIVE or UNAUTHORIZED claims "
            "(e.g. fake 'official/licensed' brand claims, exaggerated guarantees). "
            "Return JSON only: {\"riskScore\": <0..1>, \"flags\": [<short strings>], "
            "\"deceptive\": <true|false>}.\n\nListing:\n" + text
        )
        res = gemini.generate(prompt, temperature=temperature)
        if res["ok"] and res["text"]:
            parsed = _parse_json(res["text"])
            if parsed and "riskScore" in parsed:
                parsed.setdefault("flags", [])
                parsed["deceptive"] = bool(parsed.get("deceptive", parsed["riskScore"] >= 0.5))
                parsed["method"] = "gemini"
                return parsed
    return {**_semantic_heuristic(name, description), "method": "heuristic"}


def _semantic_heuristic(name, description):
    txt = f"{name or ''} {description or ''}"
    flags = []
    if _CLAIM.search(txt):
        flags.append("Unverified Authenticity Claim")
    if _HYPE.search(txt):
        flags.append("Exaggerated Claim")
    return {"riskScore": min(1.0, len(flags) * 0.35), "flags": flags, "deceptive": len(flags) > 0}


# ---------------- parsing helpers ----------------
def _parse_labeled(text):
    out = {"classification": "", "justification": "", "confidence": ""}
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("classification:"):
            out["classification"] = line.split(":", 1)[1].strip().lower()
        elif low.startswith("justification:"):
            out["justification"] = line.split(":", 1)[1].strip()
        elif low.startswith("confidence:"):
            m = re.search(r"\d+", line.split(":", 1)[1])
            out["confidence"] = int(m.group()) if m else ""
    return out


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d.]+", str(v or ""))
    val = float(m.group()) if m else 0.0
    return val / 100 if "%" in str(v) else val


if __name__ == "__main__":
    print(classify_seller("62%", "4.9/5", ["Amazing product, highly recommend!", "Love it works perfectly"]))
    print(validate_semantic("POMA Air Shoes 100% Official", "Genuine licensed authentic, best price guaranteed!"))
