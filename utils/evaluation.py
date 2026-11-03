"""
SentimentIQ Flask — Evaluation Utilities
Reads evaluation_results.json and returns formatted metrics for templates.
"""

import os
import sys
import json
import logging

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import EVAL_PATH, LABEL_NAMES

log = logging.getLogger(__name__)


def load_evaluation() -> dict | None:
    """Return evaluation results dict, or None if not yet trained."""
    if not os.path.exists(EVAL_PATH):
        return None
    try:
        with open(EVAL_PATH) as f:
            return json.load(f)
    except Exception as exc:
        log.error("Could not read evaluation results: %s", exc)
        return None


def format_metrics(results: dict) -> dict:
    """Return a UI-friendly version of evaluation metrics."""
    if results is None:
        return {}

    per_class = results.get("per_class", {})
    cm        = results.get("confusion_matrix", [])
    label_order = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]

    return {
        "accuracy":    results.get("accuracy", 0),
        "macro_f1":    results.get("macro_f1", 0),
        "cv_mean_f1":  results.get("cv_mean_f1", 0),
        "cv_std_f1":   results.get("cv_std_f1", 0),
        "n_train":     results.get("n_train", 0),
        "n_test":      results.get("n_test", 0),
        "model":       results.get("model", ""),
        "vocab_size":  results.get("tfidf_vocab_size", 0),
        "training_time": results.get("training_time_sec", 0),
        "per_class":   per_class,
        "label_order": label_order,
        "confusion_matrix":   cm,
    }
