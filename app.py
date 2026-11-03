"""
SentimentIQ Flask — Main Application
Run locally : python app.py  →  http://127.0.0.1:5000
Render/Cloud: gunicorn app:app
"""

import os
import sys
import io
import json
import uuid
import logging
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, abort,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from config.config import (
    SECRET_KEY, DEBUG, MAX_UPLOAD_MB,
    ALLOWED_EXTENSIONS, LABEL_NAMES,
    UPLOAD_FOLDER, RESULTS_FOLDER,
)
from models.predict   import predict, predict_batch
from utils.evaluation import load_evaluation, format_metrics
from utils.analytics  import record_prediction, record_batch, get_analytics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key                   = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _model_ready() -> bool:
    from config.config import MODEL_PATH
    return os.path.exists(MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html",
                           analytics=get_analytics(),
                           model_ready=_model_ready())

@app.route("/analyze")
def analyze():
    return render_template("analyze.html", model_ready=_model_ready())

@app.route("/batch")
def batch():
    return render_template("batch_analysis.html", model_ready=_model_ready())

@app.route("/performance")
def performance():
    raw     = load_evaluation()
    metrics = format_metrics(raw)
    return render_template("model_performance.html", metrics=metrics, raw=raw)

# ── API: single prediction ─────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    if not _model_ready():
        return jsonify({"error": "Model not trained yet."}), 503
    try:
        result = predict(text)
        record_prediction(result, source="single")
        return jsonify(result)
    except Exception as exc:
        log.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500

# ── API: batch prediction ──────────────────────────────────────────────────────
@app.route("/batch-predict", methods=["POST"])
def api_batch_predict():
    if not _model_ready():
        return jsonify({"error": "Model not trained yet."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file or not _allowed_file(file.filename):
        return jsonify({"error": "Please upload a valid .csv file."}), 400
    text_col = request.form.get("text_column", "text").strip()
    try:
        content = file.read().decode("utf-8", errors="replace")
        df      = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400
    if text_col not in df.columns:
        return jsonify({"error": f"Column '{text_col}' not found.",
                        "columns": df.columns.tolist()}), 400
    texts = df[text_col].fillna("").astype(str).tolist()
    if not texts:
        return jsonify({"error": "CSV has no rows."}), 400
    try:
        results = predict_batch(texts)
        record_batch(results)
    except Exception as exc:
        log.exception("Batch prediction error")
        return jsonify({"error": str(exc)}), 500
    df["predicted_sentiment"] = [r["sentiment"]  for r in results]
    df["confidence"]          = [r["confidence"] for r in results]
    df["neg_prob"]            = [r["probabilities"].get("Negative", 0) for r in results]
    df["neu_prob"]            = [r["probabilities"].get("Neutral",  0) for r in results]
    df["pos_prob"]            = [r["probabilities"].get("Positive", 0) for r in results]
    out_name = f"results_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(os.path.join(RESULTS_FOLDER, out_name), index=False)
    counts  = df["predicted_sentiment"].value_counts().to_dict()
    preview = [{"text": row[text_col][:120],
                "sentiment": row["predicted_sentiment"],
                "confidence": f'{row["confidence"]*100:.1f}%'}
               for _, row in df.head(50).iterrows()]
    return jsonify({"total": len(results), "counts": counts,
                    "preview": preview, "download_file": out_name})

# ── API: analytics ─────────────────────────────────────────────────────────────
@app.route("/analytics")
def api_analytics():
    return jsonify(get_analytics())

# ── Download CSV ───────────────────────────────────────────────────────────────
@app.route("/download-results/<filename>")
def download_results(filename):
    safe = os.path.basename(filename)
    path = os.path.join(RESULTS_FOLDER, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(RESULTS_FOLDER, safe, as_attachment=True)

# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_UPLOAD_MB} MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500

# ── Local dev ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _model_ready():
        log.warning("No model found. Run: python models/train_model.py")
    app.run(host="127.0.0.1", port=5000, debug=True)
