"""
Vercel serverless entry point for SentimentIQ Flask.
"""
import sys
import os

# ── Resolve project root ──────────────────────────────────────────────────────
_API_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_API_DIR)
sys.path.insert(0, _ROOT_DIR)

# ── Set writable /tmp paths BEFORE any app imports ────────────────────────────
os.environ["UPLOAD_FOLDER_OVERRIDE"]  = "/tmp/siq_uploads"
os.environ["RESULTS_FOLDER_OVERRIDE"] = "/tmp/siq_results"
os.environ["ANALYTICS_OVERRIDE"]      = "/tmp/siq_analytics.json"

# ── Pre-download NLTK data to /tmp (Vercel fs is read-only except /tmp) ───────
import nltk
_NLTK_DIR = "/tmp/nltk_data"
os.makedirs(_NLTK_DIR, exist_ok=True)
if _NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DIR)
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, download_dir=_NLTK_DIR, quiet=True)
    except Exception:
        pass

# ── Create writable dirs ──────────────────────────────────────────────────────
os.makedirs("/tmp/siq_uploads",  exist_ok=True)
os.makedirs("/tmp/siq_results",  exist_ok=True)

# ── Import Flask app (Vercel looks for `app` variable) ────────────────────────
from app import app
