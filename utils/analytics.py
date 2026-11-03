"""
SentimentIQ Flask — In-session Analytics
"""
import os, sys, json, logging, threading
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import ANALYTICS_PATH, LABEL_NAMES

log      = logging.getLogger(__name__)
_lock    = threading.Lock()
MAX_RECENT = 20


def _empty_store() -> dict:
    return {"total_analyzed": 0,
            "counts": {"Negative": 0, "Neutral": 0, "Positive": 0},
            "recent": []}


def _load() -> dict:
    if os.path.exists(ANALYTICS_PATH):
        try:
            with open(ANALYTICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return _empty_store()


def _save(store: dict):
    os.makedirs(os.path.dirname(ANALYTICS_PATH), exist_ok=True)
    with open(ANALYTICS_PATH, "w") as f:
        json.dump(store, f, indent=2)


def record_prediction(result: dict, source: str = "single"):
    with _lock:
        store = _load()
        s = result.get("sentiment", "Unknown")
        store["total_analyzed"] += 1
        store["counts"][s] = store["counts"].get(s, 0) + 1
        store["recent"].insert(0, {
            "text":       result.get("original_text", "")[:120],
            "sentiment":  s,
            "confidence": result.get("confidence", 0),
            "timestamp":  datetime.utcnow().isoformat(),
            "source":     source,
        })
        store["recent"] = store["recent"][:MAX_RECENT]
        _save(store)


def record_batch(results: list):
    with _lock:
        store = _load()
        for r in results:
            s = r.get("sentiment", "Unknown")
            store["total_analyzed"] += 1
            store["counts"][s] = store["counts"].get(s, 0) + 1
        _save(store)


def get_analytics() -> dict:
    with _lock:
        return _load()


def reset_analytics():
    with _lock:
        _save(_empty_store())
