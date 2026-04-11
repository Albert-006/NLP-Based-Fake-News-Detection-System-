import os
import pickle
import secrets
import logging
from datetime import datetime
from pathlib import Path

import joblib
import nltk
import requests
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
SIGHTENGINE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"
IMAGE_API_TIMEOUT = 8

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

UPLOAD_DIR.mkdir(exist_ok=True)
NLTK_DATA_DIR.mkdir(exist_ok=True)
nltk.data.path.append(str(NLTK_DATA_DIR))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(24))
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

db = SQLAlchemy(app)
http = requests.Session()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

TEXT_CLEAN_RE = re.compile("[^a-zA-Z]")
STEMMER = PorterStemmer()
STOP_WORDS = set()

_model = None
_vectorizer = None
_model_error = None


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.Text, nullable=True)
    prediction = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    image_result = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


def ensure_stopwords_available():
    try:
        stopwords.words("english")
    except Exception:
        app.logger.warning("NLTK stopwords not found locally. Downloading once to %s", NLTK_DATA_DIR)
        nltk.download("stopwords", download_dir=str(NLTK_DATA_DIR), quiet=True)
        stopwords.words("english")


def clean_text(text):
    text = TEXT_CLEAN_RE.sub(' ', str(text))
    text = text.lower().split()
    text = [STEMMER.stem(word) for word in text if word not in STOP_WORDS]
    return ' '.join(text)


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def load_ml_assets(force_retry=False):
    global _model, _vectorizer, _model_error

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if _model_error and not force_retry:
        raise RuntimeError(_model_error)

    model_path = MODELS_DIR / "model.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"

    if not model_path.exists() or not vectorizer_path.exists():
        _model_error = (
            "Model assets are missing. Add model.pkl and vectorizer.pkl to the models directory."
        )
        app.logger.error(_model_error)
        raise RuntimeError(_model_error)

    try:
        app.logger.info("Loading model assets from %s", MODELS_DIR)
        try:
            _model = joblib.load(model_path)
            _vectorizer = joblib.load(vectorizer_path)
            app.logger.info("Model assets loaded with joblib")
        except Exception:
            app.logger.warning("Joblib load failed, falling back to pickle", exc_info=True)
            with model_path.open("rb") as model_file:
                _model = pickle.load(model_file)
            with vectorizer_path.open("rb") as vectorizer_file:
                _vectorizer = pickle.load(vectorizer_file)
            app.logger.info("Model assets loaded with pickle")
        _model_error = None
        print("Model loaded:", _model is not None)
        print("Vectorizer loaded:", _vectorizer is not None)
        app.logger.info("Model loaded successfully")
    except Exception as exc:
        _model_error = (
            "Model assets could not be loaded. Verify that model.pkl and vectorizer.pkl are valid "
            "scikit-learn artifacts."
        )
        app.logger.exception("Model loading failed")
        raise RuntimeError(_model_error) from exc

    return _model, _vectorizer


def warm_up_dependencies():
    global STOP_WORDS

    app.logger.info("Starting dependency warm-up")
    ensure_stopwords_available()
    STOP_WORDS = set(stopwords.words("english"))
    app.logger.info("Stopwords loaded: %s", bool(STOP_WORDS))
    try:
        load_ml_assets(force_retry=True)
    except RuntimeError:
        app.logger.warning("Model warm-up skipped because assets are not ready")


def analyze_news_text(text):
    app.logger.info("Step 1: Text request received")
    model, vectorizer = load_ml_assets(force_retry=False)
    processed_text = clean_text(text)
    app.logger.info("Step 2: Preprocessing done")
    vector = vectorizer.transform([processed_text])
    app.logger.info("Step 3: Vectorization done")
    prediction_value = int(model.predict(vector)[0])
    confidence = float(model.predict_proba(vector).max() * 100)
    app.logger.info("Step 4: Prediction done")

    label = "FAKE NEWS" if prediction_value == 1 else "REAL NEWS"
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "processed_text": processed_text,
        "prediction_value": prediction_value,
    }


def parse_image_response(payload):
    type_data = payload.get("type", {}) or {}
    summary_data = payload.get("summary", {}) or {}
    ai_score = float(
        type_data.get("ai_generated", summary_data.get("ai_generated", 0.0)) or 0.0
    )
    label = "AI-generated" if ai_score >= 0.5 else "Real"
    return {
        "label": label,
        "confidence": round(ai_score * 100, 2),
        "raw": payload,
    }


def analyze_image(filepath):
    app.logger.info("Step 1: Image request received")
    api_user = os.getenv("SIGHTENGINE_API_USER")
    api_secret = os.getenv("SIGHTENGINE_API_SECRET")

    if not api_user or not api_secret:
        raise RuntimeError(
            "Sightengine credentials are not configured. Set SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET."
        )

    with open(filepath, "rb") as image_file:
        response = http.post(
            SIGHTENGINE_ENDPOINT,
            files={"media": image_file},
            data={
                "models": "genai",
                "api_user": api_user,
                "api_secret": api_secret,
            },
            timeout=IMAGE_API_TIMEOUT,
        )

    app.logger.info("Step 2: API response received")
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        message = payload.get("error", {}).get("message", "Sightengine analysis failed.")
        raise RuntimeError(message)

    app.logger.info("Step 3: Image response parsed")
    return parse_image_response(payload)


def save_history_entry(*, input_text=None, prediction=None, confidence=None, image_result=None):
    entry = History(
        input_text=input_text,
        prediction=prediction,
        confidence=confidence,
        image_result=image_result,
    )
    db.session.add(entry)
    db.session.commit()


def latest_history(limit=5):
    return History.query.order_by(History.timestamp.desc()).limit(limit).all()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template(
        "dashboard.html",
        text_result=None,
        image_result=None,
        recent_history=latest_history(),
    )


@app.route("/analyze/text", methods=["POST"])
def analyze_text_route():
    app.logger.info("Text analysis route hit")
    text = request.form.get("news_text", "").strip()
    text_result = None
    image_result = None

    if not text:
        flash("Please enter some news content before analyzing.", "error")
    else:
        try:
            text_result = analyze_news_text(text)
            save_history_entry(
                input_text=text,
                prediction=text_result["label"],
                confidence=text_result["confidence"],
            )
            flash("Text analysis completed successfully.", "success")
            app.logger.info("Step 5: Text response sent")
        except Exception as exc:
            app.logger.exception("Text analysis failed")
            flash(str(exc), "error")

    return render_template(
        "dashboard.html",
        text_result=text_result,
        image_result=image_result,
        recent_history=latest_history(),
        submitted_text=text,
    )


@app.route("/analyze/image", methods=["POST"])
def analyze_image_route():
    app.logger.info("Image analysis route hit")
    uploaded_file = request.files.get("image_file")
    text_result = None
    image_result = None

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose an image to analyze.", "error")
        return render_template(
            "dashboard.html",
            text_result=text_result,
            image_result=image_result,
            recent_history=latest_history(),
        )

    if not allowed_image(uploaded_file.filename):
        flash("Invalid image format. Please upload PNG, JPG, JPEG, or WEBP.", "error")
        return render_template(
            "dashboard.html",
            text_result=text_result,
            image_result=image_result,
            recent_history=latest_history(),
        )

    filename = secure_filename(uploaded_file.filename)
    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{filename}"
    filepath = UPLOAD_DIR / unique_name
    uploaded_file.save(filepath)
    app.logger.info("Image saved to %s", filepath)

    try:
        image_result = analyze_image(filepath)
        save_history_entry(
            image_result=image_result["label"],
            confidence=image_result["confidence"],
        )
        flash("Image analysis completed successfully.", "success")
        app.logger.info("Step 5: Image response sent")
    except requests.Timeout:
        app.logger.exception("Sightengine timeout")
        image_result = {
            "label": "Unavailable",
            "confidence": 0.0,
            "raw": None,
        }
        flash("Image analysis timed out after 8 seconds. Please try again.", "error")
    except requests.RequestException:
        app.logger.exception("Sightengine request failed")
        image_result = {
            "label": "Unavailable",
            "confidence": 0.0,
            "raw": None,
        }
        flash("Image analysis failed because the Sightengine API could not be reached.", "error")
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        flash(str(exc), "error")
    finally:
        try:
            filepath.unlink(missing_ok=True)
        except Exception:
            app.logger.warning("Failed to remove temporary upload %s", filepath)

    return render_template(
        "dashboard.html",
        text_result=text_result,
        image_result=image_result,
        recent_history=latest_history(),
    )


@app.route("/history")
def history():
    entries = History.query.order_by(History.timestamp.desc()).all()
    return render_template("history.html", entries=entries)


@app.context_processor
def inject_now():
    return {"current_year": datetime.utcnow().year}


@app.errorhandler(413)
def file_too_large(_error):
    flash("The uploaded file is too large. Please keep images under 5 MB.", "error")
    return redirect(url_for("dashboard"))


with app.app_context():
    db.create_all()
    warm_up_dependencies()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
