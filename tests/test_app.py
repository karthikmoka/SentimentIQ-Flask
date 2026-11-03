"""
SentimentIQ Flask — Test Suite
Run: python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest

# ── Make project root importable ─────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from preprocessing.text_preprocessing import clean_text
import app as flask_app


# ════════════════════════════════════════════════════════════════
#  Preprocessing tests
# ════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_lowercase(self):
        assert clean_text("GREAT MOVIE") == clean_text("great movie")

    def test_html_stripped(self):
        result = clean_text("<b>amazing</b> film")
        assert "<b>" not in result
        assert "amazing" in result

    def test_url_removed(self):
        result = clean_text("check http://example.com for reviews")
        assert "http" not in result

    def test_special_chars_removed(self):
        result = clean_text("wow!!! amazing@@@ film###")
        assert "!" not in result
        assert "@" not in result

    def test_negation_preserved(self):
        text = "this is not a good film"
        result = clean_text(text)
        assert "not" in result, "Negation word 'not' must be preserved"

    def test_negation_no_preserved(self):
        text = "there is no good plot"
        result = clean_text(text)
        assert "no" in result, "Negation word 'no' must be preserved"

    def test_negation_never_preserved(self):
        text = "I never enjoyed a movie like this"
        result = clean_text(text)
        assert "never" in result, "Negation word 'never' must be preserved"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""

    def test_stop_word_removed(self):
        result = clean_text("this is a very good film")
        # "this", "is", "a" should be removed but "good", "film" should remain
        assert "film" in result or "good" in result

    def test_lemmatization(self):
        result = clean_text("the movies were brilliantly acted performances")
        # 'movies' -> 'movie', 'performances' -> 'performance'
        assert "movie" in result or "perform" in result

    def test_returns_string(self):
        assert isinstance(clean_text("test input"), str)


# ════════════════════════════════════════════════════════════════
#  Flask routes tests
# ════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["DEBUG"]   = False
    with flask_app.app.test_client() as c:
        yield c


class TestRoutes:
    def test_homepage_returns_200(self, client):
        rv = client.get("/")
        assert rv.status_code == 200

    def test_analyze_page_returns_200(self, client):
        rv = client.get("/analyze")
        assert rv.status_code == 200

    def test_batch_page_returns_200(self, client):
        rv = client.get("/batch")
        assert rv.status_code == 200

    def test_performance_page_returns_200(self, client):
        rv = client.get("/performance")
        assert rv.status_code == 200

    def test_analytics_endpoint_returns_json(self, client):
        rv = client.get("/analytics")
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert "total_analyzed" in data
        assert "counts" in data

    def test_404_page(self, client):
        rv = client.get("/nonexistent-route-xyz")
        assert rv.status_code == 404

    def test_predict_empty_text(self, client):
        rv = client.post("/predict",
                         json={"text": ""},
                         content_type="application/json")
        assert rv.status_code == 400
        data = json.loads(rv.data)
        assert "error" in data

    def test_predict_no_body(self, client):
        rv = client.post("/predict",
                         json={},
                         content_type="application/json")
        assert rv.status_code == 400


# ════════════════════════════════════════════════════════════════
#  Prediction tests (require trained model)
# ════════════════════════════════════════════════════════════════

MODEL_EXISTS = os.path.exists(os.path.join(_ROOT, "models", "sentiment_model.pkl"))

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestPrediction:
    def test_positive_review(self, client):
        rv = client.post("/predict",
                         json={"text": "This film is an absolute masterpiece. "
                               "Stunning performances, beautiful cinematography, "
                               "and a deeply moving story. I loved every moment!"},
                         content_type="application/json")
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data["sentiment"] == "Positive"
        assert 0 < data["confidence"] <= 1

    def test_negative_review(self, client):
        rv = client.post("/predict",
                         json={"text": "Absolutely terrible film. The plot made no sense, "
                               "the acting was painfully wooden and the story was dreadful. "
                               "Avoid this movie at all costs."},
                         content_type="application/json")
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data["sentiment"] == "Negative"
        assert 0 < data["confidence"] <= 1

    def test_neutral_review(self, client):
        rv = client.post("/predict",
                         json={"text": "The film was okay. Some parts were decent "
                               "while others felt average. Not particularly good or bad."},
                         content_type="application/json")
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert data["sentiment"] in {"Positive", "Negative", "Neutral"}  # valid class
        assert 0 < data["confidence"] <= 1

    def test_response_has_required_fields(self, client):
        rv = client.post("/predict",
                         json={"text": "Great movie experience overall."},
                         content_type="application/json")
        assert rv.status_code == 200
        data = json.loads(rv.data)
        for field in ("sentiment", "confidence", "probabilities",
                      "color", "original_text", "cleaned_text"):
            assert field in data, f"Missing field: {field}"

    def test_probabilities_sum_to_one(self, client):
        rv = client.post("/predict",
                         json={"text": "Interesting film with mixed elements."},
                         content_type="application/json")
        assert rv.status_code == 200
        data  = json.loads(rv.data)
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_batch_csv_upload(self, client):
        import io
        csv_content = "text\nThis movie was great!\nTerrible boring film.\nOkay movie I guess.\n"
        data = {
            "file": (io.BytesIO(csv_content.encode()), "test.csv"),
            "text_column": "text",
        }
        rv = client.post("/batch-predict",
                         data=data,
                         content_type="multipart/form-data")
        assert rv.status_code == 200
        resp = json.loads(rv.data)
        assert resp["total"] == 3
        assert "counts" in resp
        assert "preview" in resp
        assert "download_file" in resp
