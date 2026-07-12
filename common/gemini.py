"""gemini.py — robust Google Gemini REST client (no extra SDK; uses requests).

Fixes the issues found in the original scripts:
  * model name 'gemini-2.0-flash-exp' (deprecated) -> configurable, defaults to 'gemini-2.0-flash'
  * brittle response parsing of candidates[].content.parts[].text
"""
from __future__ import annotations
import base64
import os
import mimetypes

import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "20"))


def api_key():
    return os.environ.get("GEMINI_API_KEY")


def available():
    return bool(api_key())


def _extract_text(data: dict) -> str:
    try:
        cand = data["candidates"][0]
        content = cand.get("content", {})
        if isinstance(content, dict):
            parts = content.get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        return str(content)
    except (KeyError, IndexError, TypeError):
        return ""


def generate(prompt: str, images: list[str] | None = None,
             temperature: float = 0.0, model: str | None = None) -> dict:
    """Call Gemini generateContent. Returns {ok, text, error}."""
    key = api_key()
    if not key:
        return {"ok": False, "text": "", "error": "no GEMINI_API_KEY"}

    model = model or GEMINI_MODEL
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")

    parts: list[dict] = [{"text": prompt}]
    for img in (images or []):
        try:
            with open(img, "rb") as f:
                raw = f.read()
            mime = mimetypes.guess_type(img)[0] or "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime,
                                          "data": base64.b64encode(raw).decode()}})
        except OSError:
            continue

    payload = {"contents": [{"parts": parts}],
               "generationConfig": {"temperature": temperature}}
    try:
        r = requests.post(url, json=payload, timeout=_TIMEOUT)
        if r.status_code != 200:
            return {"ok": False, "text": "", "error": f"http {r.status_code}: {r.text[:200]}"}
        return {"ok": True, "text": _extract_text(r.json()), "error": None}
    except requests.RequestException as e:
        return {"ok": False, "text": "", "error": str(e)}
