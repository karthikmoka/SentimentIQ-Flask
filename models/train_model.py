"""
SentimentIQ Flask — Model Training Script

Trains TF-IDF + Logistic Regression on the SST-5 (3-class) dataset.
Saves:
  models/sentiment_model.pkl    — sklearn Pipeline (vectorizer + classifier)
  models/evaluation_results.json — full metrics dict
"""

import os
import sys
import json
import time
import logging

import numpy  as np
import pandas as pd
import joblib

from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Resolve project root regardless of CWD ───────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config.config import (
    DATASET_DIR, MODEL_PATH, EVAL_PATH,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    LR_C, LR_CLASS_WEIGHT, LR_MAX_ITER, LR_SOLVER, LR_MULTI_CLASS,
    TEST_SIZE, RANDOM_STATE, LABEL_NAMES,
)
from preprocessing.text_preprocessing import preprocess_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return (train_df, test_df)."""
    train_path = os.path.join(DATASET_DIR, "train.csv")
    test_path  = os.path.join(DATASET_DIR, "test.csv")

    log.info("Loading train data from %s", train_path)
    train_df = pd.read_csv(train_path)
    log.info("Loading test  data from %s", test_path)
    test_df  = pd.read_csv(test_path)

    # Keep only text + label columns
    for df in (train_df, test_df):
        assert "text"  in df.columns, "Missing 'text' column"
        assert "label" in df.columns, "Missing 'label' column"

    # Drop rows with missing text or label
    train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)

    log.info("Train: %d rows  |  Test: %d rows", len(train_df), len(test_df))
    log.info("Label distribution (train):\n%s", train_df["label"].value_counts().to_string())
    return train_df, test_df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    log.info("Preprocessing text …")
    X_train = preprocess_series(train_df["text"])
    y_train = train_df["label"].astype(int).values
    X_test  = preprocess_series(test_df["text"])
    y_test  = test_df["label"].astype(int).values
    log.info("Preprocessing done.")
    return X_train, y_train, X_test, y_test


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features   = TFIDF_MAX_FEATURES,
        ngram_range    = TFIDF_NGRAM_RANGE,
        sublinear_tf   = TFIDF_SUBLINEAR_TF,
        strip_accents  = "unicode",
        analyzer       = "word",
        token_pattern  = r"\b[a-zA-Z']{2,}\b",
        min_df         = 2,
    )
    classifier = LogisticRegression(
        C            = LR_C,
        class_weight = LR_CLASS_WEIGHT,
        max_iter     = LR_MAX_ITER,
        solver       = LR_SOLVER,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def evaluate(pipeline, X_train, y_train, X_test, y_test) -> dict:
    log.info("Evaluating …")
    y_pred      = pipeline.predict(X_test)
    y_proba     = pipeline.predict_proba(X_test)
    acc         = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True,
    )
    cm          = confusion_matrix(y_test, y_pred).tolist()
    labels      = sorted(LABEL_NAMES.keys())
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None
    )
    macro_f1    = float(np.mean(f1))

    # 3-fold cross-val on training set
    log.info("Running 3-fold cross-validation on training set …")
    cv_scores   = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)

    results = {
        "accuracy":          round(float(acc), 4),
        "macro_f1":          round(macro_f1, 4),
        "cv_mean_f1":        round(float(cv_scores.mean()), 4),
        "cv_std_f1":         round(float(cv_scores.std()), 4),
        "classification_report": report_dict,
        "confusion_matrix":  cm,
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(float(prec[idx]), 4),
                "recall":    round(float(rec[idx]),  4),
                "f1":        round(float(f1[idx]),   4),
            }
            for idx, i in enumerate(labels)
        },
        "n_train": int(len(y_train)),
        "n_test":  int(len(y_test)),
        "model":   "TF-IDF + Logistic Regression",
        "tfidf_vocab_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
    }

    log.info("Test accuracy : %.4f", acc)
    log.info("Test Macro-F1 : %.4f", macro_f1)
    log.info("CV Macro-F1   : %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    print("\n", classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
    ))
    return results


def train():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

    log.info("Building pipeline …")
    pipeline = build_pipeline()

    log.info("Training …")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    log.info("Training complete in %.1fs", elapsed)

    results = evaluate(pipeline, X_train, y_train, X_test, y_test)
    results["training_time_sec"] = round(elapsed, 2)

    # Save model
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    log.info("Model saved → %s", MODEL_PATH)

    # Save evaluation
    with open(EVAL_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Metrics saved → %s", EVAL_PATH)

    return pipeline, results


if __name__ == "__main__":
    train()
