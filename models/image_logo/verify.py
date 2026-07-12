"""verify.py — Image / Logo Counterfeit Verification.

Two complementary signals:
  1. Gemini vision: holistic counterfeit assessment of the product image vs the
     claimed brand/product name (primary; works with just an API key).
  2. Fake-logo CNN: the project's trained Keras model (fake_logo_detector.h5),
     loaded lazily via TensorFlow if available (secondary).
Falls back to a neutral heuristic if neither signal is available.
"""
from __future__ import annotations
import json
import os
import re
import threading
from pathlib import Path

from common import gemini

FAKE_LOGO_H5 = os.environ.get(
    "FAKE_LOGO_H5",
    r"D:\Amazon HackOn S5\Amazon HackOn Models\Fake Logo Detection\Fake-Logo-Detection-using-Djnago\fake_logo_detector.h5",
)
IMG_SIZE = (150, 150)

_cnn = {}
_lock = threading.Lock()


def _load_cnn():
    if _cnn:
        return _cnn.get("model")
    with _lock:
        if _cnn:
            return _cnn.get("model")
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf  # noqa
            if not Path(FAKE_LOGO_H5).exists():
                raise FileNotFoundError(FAKE_LOGO_H5)
            _cnn["model"] = tf.keras.models.load_model(FAKE_LOGO_H5)
            _cnn["tf"] = tf
            print(f"[image] fake-logo CNN loaded from {FAKE_LOGO_H5}")
        except Exception as e:
            print(f"[image] CNN unavailable ({e}); using Gemini/heuristic only")
            _cnn["model"] = None
            _cnn["error"] = str(e)
        return _cnn.get("model")


def _cnn_fake_prob(image_path: str):
    model = _load_cnn()
    if model is None:
        return None
    try:
        tf = _cnn["tf"]
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        arr = arr.reshape((1, *IMG_SIZE, 3))
        # flow_from_directory assigns classes alphabetically: fake=0, real=1
        p_real = float(model.predict(arr, verbose=0)[0][0])
        return max(0.0, min(1.0, 1.0 - p_real))
    except Exception as e:
        print(f"[image] CNN predict error: {e}")
        return None


def _gemini_fake_prob(image_path: str, product_name: str):
    if not gemini.available():
        return None
    prompt = (
        f"You are a counterfeit-detection expert. The seller lists this image as "
        f"'{product_name}'. Assess if the product/logo looks counterfeit, misbranded, "
        f"or inconsistent with the claimed brand. Return JSON only: "
        f"{{\"fakeProbability\": <0..1>, \"label\": <short>, \"flags\": [<short strings>]}}."
    )
    res = gemini.generate(prompt, images=[image_path])
    if not res["ok"] or not res["text"]:
        return None
    m = re.search(r"\{.*\}", res["text"], re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group())
        return {
            "fakeProbability": float(data.get("fakeProbability", 0)),
            "label": data.get("label", "assessed"),
            "flags": data.get("flags", []),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def verify(image_paths, product_name: str = ""):
    image_paths = [p for p in (image_paths or []) if p and Path(p).exists()]
    if not image_paths:
        return {"fakeProbability": 0.1, "label": "no-image", "flags": [], "method": "none"}

    probs = []
    flags = set()
    methods = []
    for path in image_paths[:4]:
        g = _gemini_fake_prob(path, product_name)
        if g is not None:
            probs.append(g["fakeProbability"])
            for fl in g.get("flags", []):
                flags.add(fl)
            methods.append("gemini-vision")
        c = _cnn_fake_prob(path)
        if c is not None:
            probs.append(c)
            methods.append("logo-cnn")
            if c >= 0.5:
                flags.add("Logo Mismatch")

    if not probs:
        return {"fakeProbability": 0.1, "label": "unverified", "flags": [], "method": "heuristic"}

    fake_prob = max(probs)  # most-suspicious image drives the verdict
    label = "likely-counterfeit" if fake_prob >= 0.5 else "likely-authentic"
    if fake_prob >= 0.5:
        flags.add("Image Mismatch")
    return {
        "fakeProbability": round(fake_prob, 4),
        "label": label,
        "flags": sorted(flags),
        "method": "+".join(sorted(set(methods))) or "heuristic",
    }


if __name__ == "__main__":
    print(verify([], "Test Product"))
