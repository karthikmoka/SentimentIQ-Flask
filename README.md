# SentimentIQ — AI Movie Review Intelligence (Flask Edition)

> **AI-Powered Sentiment Analysis Web Application**  
> Built with Flask · TF-IDF + Logistic Regression · NLTK · Bootstrap 5 · Chart.js

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Features](#features)
6. [Dataset](#dataset)
7. [NLP Preprocessing Pipeline](#nlp-preprocessing-pipeline)
8. [Machine Learning Model](#machine-learning-model)
9. [Model Performance](#model-performance)
10. [Installation & Setup](#installation--setup)
11. [Running the Application](#running-the-application)
12. [API Reference](#api-reference)
13. [Testing](#testing)
14. [Limitations & Future Work](#limitations--future-work)
15. [Academic Context](#academic-context)

---

## Project Overview

**SentimentIQ (Flask Edition)** is a fully functional, end-to-end web application for sentiment analysis of movie reviews. Users can:

- Analyze individual review texts and receive an AI-generated sentiment prediction with confidence score
- Upload CSV files for batch sentiment prediction on hundreds of reviews
- Explore model performance metrics including accuracy, Macro-F1, confusion matrix, and per-class scores
- View an analytics dashboard tracking cumulative prediction history

The backend uses a **TF-IDF + Logistic Regression** pipeline trained on the Stanford Sentiment Treebank (SST-5) dataset, mapped to three classes: **Positive**, **Negative**, and **Neutral**.

---

## Problem Statement

Natural language text contains rich emotional signals that are difficult to quantify manually. Sentiment analysis automates the classification of opinion-bearing text into polarity classes. Applications include:

- Customer feedback monitoring
- Brand reputation tracking
- Movie/product review aggregation
- Academic research on opinion mining

This project solves the **three-class sentiment classification problem** for movie reviews.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Chart.js 4.4, Bootstrap Icons |
| **Backend**  | Python 3.10, Flask 3.1 |
| **ML / NLP** | Scikit-learn 1.7, NLTK 3.10, BeautifulSoup4 |
| **Data**     | Pandas 2.3, NumPy 2.2, Joblib 1.5 |
| **Testing**  | Pytest 9.1 |

---

## Project Structure

```
SentimentIQ_Flask/
├── app.py                         # Flask application entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── config/
│   └── config.py                  # Centralised configuration constants
│
├── preprocessing/
│   └── text_preprocessing.py      # NLTK NLP pipeline (clean_text, preprocess_series)
│
├── models/
│   ├── train_model.py             # Training script (TF-IDF + Logistic Regression)
│   ├── predict.py                 # Inference module (predict, predict_batch)
│   ├── sentiment_model.pkl        # Trained pipeline (generated after training)
│   └── evaluation_results.json   # Evaluation metrics (generated after training)
│
├── dataset/
│   └── raw/
│       ├── train.csv              # Training data (SST-5, 3-class)
│       └── test.csv               # Test data (SST-5, 3-class)
│
├── utils/
│   ├── evaluation.py              # Loads & formats evaluation metrics
│   └── analytics.py              # Persistent analytics (results/analytics.json)
│
├── templates/
│   ├── base.html                  # Shared layout (navbar, footer, CDN links)
│   ├── index.html                 # Home dashboard
│   ├── analyze.html               # Single text analysis page
│   ├── batch_analysis.html        # CSV batch upload & results
│   ├── model_performance.html     # Model evaluation dashboard
│   └── 404.html                   # 404 error page
│
├── static/
│   ├── css/style.css              # Custom stylesheet (design system)
│   └── js/script.js               # Chart.js charts + AJAX prediction logic
│
├── results/
│   └── analytics.json            # Cumulative prediction analytics (auto-created)
│
├── uploads/                       # Temporary CSV uploads (auto-cleaned)
└── tests/
    └── test_app.py                # 27 pytest tests (preprocessing + routes + predictions)
```

---

## Features

### 🏠 Home Dashboard
- Live statistics: total analyses, positive/negative/neutral counts
- Interactive Chart.js doughnut and bar charts
- Recent analysis history table
- Sentiment category cards with examples

### 🔍 Single Text Analysis
- Large textarea with character counter
- One-click quick-example buttons (Positive / Negative / Neutral)
- Real-time AJAX prediction (no page reload)
- Color-coded result badge, confidence percentage, per-class probability bars
- Session history table tracking all analyses in the current browser session

### 📊 Batch CSV Analysis
- Drag-and-drop CSV file upload
- Configurable text column name
- Predictions for every row in the CSV
- Summary statistics and downloadable results CSV
- Preview table (first 50 rows)

### 📈 Model Performance
- Test accuracy, Macro-F1, CV F1 (3-fold) metric cards
- Per-class Precision / Recall / F1 with visual progress bars
- Interactive Chart.js grouped bar chart (precision, recall, F1 per class)
- Confusion matrix heatmap table
- Model info card (training time, vocabulary size, dataset details)

---

## Dataset

**Stanford Sentiment Treebank — SST-5** (via Hugging Face `SetFit/sst5`)

| Split | Rows |
|---|---|
| Train | 8,544 |
| Test  | 2,210 |

**Five-class → Three-class mapping:**

| Original SST Label | Mapped Class |
|---|---|
| 0 — very negative  | **Negative** |
| 1 — somewhat negative | **Negative** |
| 2 — neutral        | **Neutral** |
| 3 — somewhat positive | **Positive** |
| 4 — very positive  | **Positive** |

**Class distribution (training set):**

| Class | Count |
|---|---|
| Positive | 3,610 |
| Negative | 3,310 |
| Neutral  | 1,624 |

---

## NLP Preprocessing Pipeline

Implemented in [`preprocessing/text_preprocessing.py`](preprocessing/text_preprocessing.py).

| Step | Operation |
|---|---|
| 1 | Lowercase |
| 2 | Strip HTML tags (BeautifulSoup) |
| 3 | Remove URLs (regex) |
| 4 | Remove special characters (keep letters + apostrophes) |
| 5 | Tokenize (NLTK `word_tokenize`) |
| 6 | Remove stop-words (NLTK) — **negation words preserved** |
| 7 | Lemmatize (NLTK `WordNetLemmatizer`) |

**Preserved negation words** (critical for sentiment accuracy):
`not`, `no`, `never`, `neither`, `nor`, `nobody`, `nothing`, `nowhere`, `none`, `hardly`, `barely`, `scarcely`, `doesn't`, `isn't`, `wasn't`, `aren't`, `won't`, `wouldn't`, `don't`, `didn't`, `haven't`, `hasn't`, `hadn't`, `shouldn't`, `couldn't`, `mustn't`, `cannot`, `can't`, `needn't`

> ⚠️ **Important**: The exact same preprocessing pipeline is applied at both training and inference time to prevent data leakage and ensure consistency.

---

## Machine Learning Model

**Architecture:** Scikit-learn `Pipeline` containing:
1. `TfidfVectorizer` — converts cleaned text to TF-IDF weighted feature matrix
2. `LogisticRegression` — multi-class classifier (multinomial)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| TF-IDF max features | 15,000 |
| TF-IDF n-gram range | (1, 2) — unigrams + bigrams |
| TF-IDF sublinear TF | True |
| TF-IDF min_df | 2 |
| LR C (regularisation) | 1.0 |
| LR class_weight | balanced |
| LR solver | lbfgs |
| LR max_iter | 1,000 |

**Model file:** `models/sentiment_model.pkl` (serialised with Joblib)

---

## Model Performance

Results on the SST-5 test set (2,210 examples):

| Metric | Value |
|---|---|
| Test Accuracy | **60.4%** |
| Macro F1-Score | **0.537** |
| CV Macro-F1 (3-fold) | 0.534 ± 0.014 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.70 | 0.73 | 0.71 |
| Negative | 0.69 | 0.63 | 0.66 |
| Neutral  | 0.22 | 0.25 | 0.24 |

> **Note on accuracy:** The three-class SST task is inherently harder than binary classification. The Neutral class is underrepresented and linguistically ambiguous (originally "somewhat" positive/negative reviews compressed to a single label), which depresses Neutral-class F1. The Positive and Negative classes perform well (F1 ≥ 0.66). This is a real, measured result on a held-out test set — no metric inflation.

---

## Installation & Setup

### Prerequisites
- Python 3.10 or later
- pip

### 1. Clone / download the project
```bash
cd SentimentIQ_Flask
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)
```python
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Train the model
```bash
python models/train_model.py
```
This creates:
- `models/sentiment_model.pkl` — trained pipeline
- `models/evaluation_results.json` — evaluation metrics

---

## Running the Application

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## API Reference

### `POST /predict`
Predict sentiment for a single text.

**Request body (JSON):**
```json
{ "text": "This movie was absolutely fantastic!" }
```

**Response (200 OK):**
```json
{
  "sentiment": "Positive",
  "label": 2,
  "confidence": 0.8923,
  "probabilities": { "Negative": 0.0421, "Neutral": 0.0656, "Positive": 0.8923 },
  "color": "#16A34A",
  "original_text": "This movie was absolutely fantastic!",
  "cleaned_text": "movie absolutely fantastic"
}
```

**Error responses:**
- `400` — empty or missing text
- `503` — model not yet trained

---

### `POST /batch-predict`
Batch prediction from CSV file upload.

**Form data:**
- `file` — CSV file (multipart/form-data)
- `text_column` — name of the text column (default: `text`)

**Response (200 OK):**
```json
{
  "total": 150,
  "counts": { "Positive": 80, "Negative": 50, "Neutral": 20 },
  "preview": [ { "text": "...", "sentiment": "Positive", "confidence": "89.2%" } ],
  "download_file": "results_abc12345.csv"
}
```

---

### `GET /analytics`
Returns cumulative analytics store.

```json
{
  "total_analyzed": 42,
  "counts": { "Positive": 20, "Negative": 15, "Neutral": 7 },
  "recent": [ { "text": "...", "sentiment": "Positive", "confidence": 0.89, "timestamp": "..." } ]
}
```

---

### `GET /download-results/<filename>`
Download a batch results CSV file.

---

## Testing

```bash
python -m pytest tests/ -v
```

**Test coverage:** 27 tests across three classes:

| Test Class | Tests | Scope |
|---|---|---|
| `TestPreprocessing` | 13 | HTML stripping, URL removal, negation preservation, lemmatization, edge cases |
| `TestRoutes` | 8 | All page routes (200 OK), analytics JSON, 404, empty/missing inputs |
| `TestPrediction` | 6 | Positive/negative/negative predictions, required fields, probability sum, batch upload |

All 27 tests pass on a correctly trained model.

---

## Limitations & Future Work

### Current Limitations
- **Neutral class accuracy is low** (F1 ≈ 0.24): The SST-5 neutral class represents genuinely ambiguous text, and TF-IDF features struggle to capture subtle contextual signals. This is a known challenge in the literature.
- **No context window**: TF-IDF treats text as a bag of words and cannot capture long-range dependencies or semantic meaning.
- **Domain specificity**: The model is trained on short movie review snippets. Performance on longer or domain-shifted texts (e.g., social media) may vary.

### Future Improvements
| Improvement | Expected Impact |
|---|---|
| BERT / RoBERTa fine-tuning | Significant improvement on Neutral class |
| Larger dataset (Amazon reviews) | Better generalisation |
| Aspect-level sentiment analysis | Richer analysis per product/film attribute |
| User authentication + history | Personalised dashboard |
| REST API with JWT auth | Production deployment |

---

## Academic Context

This project was built as a **B.Tech AI & ML semester project** demonstrating:

- **NLP Fundamentals**: tokenisation, stop-word removal, lemmatisation, TF-IDF
- **Machine Learning Workflow**: data loading → preprocessing → training → evaluation → inference
- **Software Engineering**: modular architecture, configuration management, error handling
- **Full-Stack Development**: Flask backend, responsive Bootstrap frontend, Chart.js visualisations
- **Testing**: 27 pytest tests covering preprocessing, routes, and ML predictions

**Dataset citation:**  
Socher, R. et al. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank." EMNLP 2013.

---

*SentimentIQ Flask Edition — Built with ❤️ using Python, Flask, and Scikit-learn*
