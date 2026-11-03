"""
SentimentIQ Flask — Configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DIR    = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_PATH       = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
EVAL_PATH        = os.path.join(BASE_DIR, "models", "evaluation_results.json")
ANALYTICS_PATH   = os.path.join(BASE_DIR, "results", "analytics.json")
UPLOAD_FOLDER    = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER   = os.path.join(BASE_DIR, "results")

# ── Model ────────────────────────────────────────────────────────────────────
LABEL_NAMES  = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS = {0: "#DC2626", 1: "#EAB308", 2: "#16A34A"}   # red / yellow / green

# TF-IDF
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_SUBLINEAR_TF = True

# Logistic Regression
LR_C             = 1.0
LR_CLASS_WEIGHT  = "balanced"
LR_MAX_ITER      = 1000
LR_SOLVER        = "lbfgs"
LR_MULTI_CLASS   = "multinomial"

# Train/test split
TEST_SIZE        = 0.20
RANDOM_STATE     = 42

# ── Upload ───────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB    = 10
ALLOWED_EXTENSIONS = {"csv"}

# ── Flask ────────────────────────────────────────────────────────────────────
SECRET_KEY = "sentimentiq-secret-2024"
DEBUG      = False
