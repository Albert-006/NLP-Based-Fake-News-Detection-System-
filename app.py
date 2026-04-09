import os
import pickle
import secrets
from datetime import datetime
from pathlib import Path

import nltk
import requests
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
SIGHTENGINE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"

UPLOAD_DIR.mkdir(exist_ok=True)

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
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

_model = None
_vectorizer = None
_model_error = None


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    history_entries = db.relationship("AnalysisHistory", backref="user", lazy=True)


class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    input_text = db.Column(db.Text, nullable=True)
    prediction = db.Column(db.String(32), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    image_result = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def ensure_stopwords_available():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def load_ml_assets():
    global _model, _vectorizer, _model_error

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if _model_error:
        raise RuntimeError(_model_error)

    model_path = MODELS_DIR / "model.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"

    if not model_path.exists() or not vectorizer_path.exists():
        _model_error = (
            "Model assets are missing. Add model.pkl and vectorizer.pkl to the models directory."
        )
        raise RuntimeError(_model_error)

    try:
        with model_path.open("rb") as model_file:
            _model = pickle.load(model_file)
        with vectorizer_path.open("rb") as vectorizer_file:
            _vectorizer = pickle.load(vectorizer_file)
    except Exception as exc:
        _model_error = f"Failed to load ML assets: {exc}"
        raise RuntimeError(_model_error) from exc

    return _model, _vectorizer


def analyze_news_text(text):
    ensure_stopwords_available()
    model, vectorizer = load_ml_assets()
    processed_text = clean_text(text)
    vector = vectorizer.transform([processed_text])
    prediction_value = int(model.predict(vector)[0])
    confidence = float(model.predict_proba(vector).max() * 100)

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
    api_user = os.getenv("SIGHTENGINE_API_USER")
    api_secret = os.getenv("SIGHTENGINE_API_SECRET")

    if not api_user or not api_secret:
        raise RuntimeError(
            "Sightengine credentials are not configured. Set SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET."
        )

    with open(filepath, "rb") as image_file:
        response = requests.post(
            SIGHTENGINE_ENDPOINT,
            files={"media": image_file},
            data={
                "models": "genai",
                "api_user": api_user,
                "api_secret": api_secret,
            },
            timeout=30,
        )

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        message = payload.get("error", {}).get("message", "Sightengine analysis failed.")
        raise RuntimeError(message)

    return parse_image_response(payload)


def save_history_entry(*, user_id, input_text=None, prediction=None, confidence=None, image_result=None):
    entry = AnalysisHistory(
        user_id=user_id,
        input_text=input_text,
        prediction=prediction,
        confidence=confidence,
        image_result=image_result,
    )
    db.session.add(entry)
    db.session.commit()


def latest_history(limit=5):
    if not current_user.is_authenticated:
        return []
    return (
        AnalysisHistory.query.filter_by(user_id=current_user.id)
        .order_by(AnalysisHistory.timestamp.desc())
        .limit(limit)
        .all()
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("Please fill in all required fields.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif User.query.filter((User.email == email) | (User.username == username)).first():
            flash("An account with that email or username already exists.", "error")
        else:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
            )
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash("Account created successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Welcome back.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        text_result=None,
        image_result=None,
        recent_history=latest_history(),
    )


@app.route("/analyze/text", methods=["POST"])
@login_required
def analyze_text_route():
    text = request.form.get("news_text", "").strip()
    text_result = None
    image_result = None

    if not text:
        flash("Please enter some news content before analyzing.", "error")
    else:
        try:
            text_result = analyze_news_text(text)
            save_history_entry(
                user_id=current_user.id,
                input_text=text,
                prediction=text_result["label"],
                confidence=text_result["confidence"],
            )
            flash("Text analysis completed successfully.", "success")
        except Exception as exc:
            flash(str(exc), "error")

    return render_template(
        "dashboard.html",
        text_result=text_result,
        image_result=image_result,
        recent_history=latest_history(),
        submitted_text=text,
    )


@app.route("/analyze/image", methods=["POST"])
@login_required
def analyze_image_route():
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

    try:
        image_result = analyze_image(filepath)
        save_history_entry(
            user_id=current_user.id,
            image_result=image_result["label"],
            confidence=image_result["confidence"],
        )
        flash("Image analysis completed successfully.", "success")
    except requests.RequestException:
        flash("Image analysis failed because the Sightengine API could not be reached.", "error")
    except Exception as exc:
        flash(str(exc), "error")
    finally:
        try:
            filepath.unlink(missing_ok=True)
        except Exception:
            pass

    return render_template(
        "dashboard.html",
        text_result=text_result,
        image_result=image_result,
        recent_history=latest_history(),
    )


@app.route("/history")
@login_required
def history():
    entries = (
        AnalysisHistory.query.filter_by(user_id=current_user.id)
        .order_by(AnalysisHistory.timestamp.desc())
        .all()
    )
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
