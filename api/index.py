"""
Vercel serverless entry point for SentimentIQ Flask.
"""
import sys
import os
import traceback

# ── Resolve paths ─────────────────────────────────────────────────────────────
_API_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_API_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _API_DIR)

# ── Writable /tmp paths (Vercel filesystem is read-only except /tmp) ──────────
os.environ["UPLOAD_FOLDER_OVERRIDE"]  = "/tmp/siq_uploads"
os.environ["RESULTS_FOLDER_OVERRIDE"] = "/tmp/siq_results"
os.environ["ANALYTICS_OVERRIDE"]      = "/tmp/siq_analytics.json"

for _d in ["/tmp/siq_uploads", "/tmp/siq_results", "/tmp/nltk_data"]:
    os.makedirs(_d, exist_ok=True)

# ── NLTK data → /tmp ──────────────────────────────────────────────────────────
try:
    import nltk
    _NLTK_DIR = "/tmp/nltk_data"
    if _NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DIR)
    for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(_pkg, download_dir=_NLTK_DIR, quiet=True)
        except Exception:
            pass
except Exception:
    pass

# ── Import Flask app ──────────────────────────────────────────────────────────
try:
    from app import app
except Exception as _e:
    # If import fails, return an informative error page
    from flask import Flask, jsonify
    app = Flask(__name__)
    _import_error = traceback.format_exc()

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return f"""
        <html><body style="font-family:monospace;padding:2rem;background:#1e1e1e;color:#f1f1f1">
        <h2 style="color:#ff6b6b">&#x26A0; Startup Error</h2>
        <p>The app failed to import. Error details:</p>
        <pre style="background:#2d2d2d;padding:1rem;border-radius:8px;overflow:auto">{_import_error}</pre>
        <p>Please check the Vercel function logs for more details.</p>
        </body></html>
        """, 500
