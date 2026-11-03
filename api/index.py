"""
Vercel serverless entry point — imports the Flask app from project root.
"""
import sys
import os

# Add project root to path so all modules resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pre-download NLTK data to /tmp before anything else imports it
import nltk
_NLTK_DIR = "/tmp/nltk_data"
os.makedirs(_NLTK_DIR, exist_ok=True)
if _NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DIR)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, download_dir=_NLTK_DIR, quiet=True)
    except Exception:
        pass

# Set environment overrides for writable paths on Vercel
os.environ["UPLOAD_FOLDER_OVERRIDE"]  = "/tmp/siq_uploads"
os.environ["RESULTS_FOLDER_OVERRIDE"] = "/tmp/siq_results"
os.environ["ANALYTICS_OVERRIDE"]      = "/tmp/siq_analytics.json"
os.makedirs("/tmp/siq_uploads",  exist_ok=True)
os.makedirs("/tmp/siq_results",  exist_ok=True)

# Import the Flask app — Vercel looks for variable named `app`
from app import app
