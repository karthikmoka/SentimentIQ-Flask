"""
SentimentIQ Flask — Prediction Module

Loads the trained pipeline and exposes predict() / predict_batch().
"""

import os
import sys
import logging
from typing import Optional

import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import MODEL_PATH, LABEL_NAMES, LABEL_COLORS
from preprocessing.text_preprocessing import clean_text

log = logging.getLogger(__name__)

_pipeline = None   # lazy-loaded singleton


def _load_model():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python models/train_model.py` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    return _pipeline


def predict(text: str) -> dict:
    """
    Predict sentiment for a single text string.

    Returns:
        {
          "sentiment":   "Positive" | "Negative" | "Neutral",
          "label":       2 | 0 | 1,
          "confidence":  0.92,          # max class probability
          "probabilities": {"Negative": 0.04, "Neutral": 0.04, "Positive": 0.92},
          "color":       "#16A34A",
          "original_text": "<raw text>",
          "cleaned_text":  "<preprocessed text>",
        }
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    pipe        = _load_model()
    cleaned     = clean_text(text)
    proba       = pipe.predict_proba([cleaned])[0]
    label_int   = int(np.argmax(proba))
    label_name  = LABEL_NAMES[label_int]
    confidence  = float(proba[label_int])

    # Build per-class probability dict (class order matches pipe.classes_)
    classes = [int(c) for c in pipe.classes_]
    probs   = {LABEL_NAMES[c]: round(float(p), 4) for c, p in zip(classes, proba)}

    return {
        "sentiment":     label_name,
        "label":         label_int,
        "confidence":    round(confidence, 4),
        "probabilities": probs,
        "color":         LABEL_COLORS[label_int],
        "original_text": text,
        "cleaned_text":  cleaned,
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Predict sentiments for a list of texts.  Returns a list of result dicts.
    """
    if not texts:
        return []

    pipe     = _load_model()
    cleaned  = [clean_text(t) for t in texts]
    probas   = pipe.predict_proba(cleaned)
    classes  = [int(c) for c in pipe.classes_]

    results = []
    for raw, clean, proba in zip(texts, cleaned, probas):
        label_int  = int(np.argmax(proba))
        label_name = LABEL_NAMES[label_int]
        probs      = {LABEL_NAMES[c]: round(float(p), 4) for c, p in zip(classes, proba)}
        results.append({
            "sentiment":     label_name,
            "label":         label_int,
            "confidence":    round(float(proba[label_int]), 4),
            "probabilities": probs,
            "color":         LABEL_COLORS[label_int],
            "original_text": raw,
            "cleaned_text":  clean,
        })
    return results
