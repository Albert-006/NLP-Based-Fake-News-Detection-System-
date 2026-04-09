# NLP-Based-Fake-News-Detection-System-

A production-ready Flask web application for detecting fake news with a custom NLP model and identifying AI-generated images with the Sightengine API. The project includes global analysis history and a modern Tailwind CSS interface suitable for portfolio presentation and cloud deployment.

## Overview

This application combines two AI-assisted verification workflows in one dashboard:

- Fake News Detection using a trained `model.pkl` and `vectorizer.pkl`
- AI Image Detection using Sightengine's `genai` model
- Public access with no login required
- Global analysis history tracking
- Responsive glassmorphism UI built with Tailwind CSS

## Features

- Custom-trained NLP fake news classifier
- Exact text preprocessing pipeline with NLTK stopwords and Porter stemming
- Confidence scoring using `model.predict_proba(vector).max() * 100`
- Sightengine image classification with confidence scoring
- Drag-and-drop image upload experience
- Loading overlays and friendly validation/error states
- SQLite persistence for users and history entries
- Render-ready deployment setup with Gunicorn

## Tech Stack

- Python
- Flask
- Flask-Login
- Flask-SQLAlchemy
- scikit-learn
- NLTK
- Requests
- Tailwind CSS (CDN)
- SQLite
- Gunicorn

## Folder Structure

```text
NLP-Based-Fake-News-Detection-System-/
|-- app.py
|-- Procfile
|-- README.md
|-- requirements.txt
|-- runtime.txt
|-- .env.example
|-- .gitignore
|-- app.db                    # created automatically after first run
|-- models/
|   |-- .gitkeep
|   |-- model.pkl             # add your trained model here
|   `-- vectorizer.pkl        # add your trained vectorizer here
|-- uploads/
|   `-- .gitkeep
|-- static/
|   |-- css/
|   |   `-- styles.css
|   `-- js/
|       `-- app.js
`-- templates/
    |-- base.html
    |-- home.html
|-- dashboard.html
`-- history.html
```

## Model Integration Details

The app implements the required preprocessing exactly before vectorization:

```python
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
```

Prediction mapping used by the app:

- `1 = FAKE NEWS`
- `0 = REAL NEWS`

## Local Setup

1. Clone the repository and move into the project directory.
2. Create a virtual environment.
3. Install dependencies.
4. Add `model.pkl` and `vectorizer.pkl` inside the `models/` directory.
5. Set environment variables.
6. Run the app.

### Windows PowerShell

```powershell
cd D:\Files\Game\NLP-Based-Fake-News-Detection-System-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SECRET_KEY="change-this-in-production"
$env:SIGHTENGINE_API_USER="your_sightengine_user"
$env:SIGHTENGINE_API_SECRET="your_sightengine_secret"
python app.py
```

### macOS / Linux

```bash
cd NLP-Based-Fake-News-Detection-System-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY="change-this-in-production"
export SIGHTENGINE_API_USER="your_sightengine_user"
export SIGHTENGINE_API_SECRET="your_sightengine_secret"
python app.py
```

The application runs on `http://127.0.0.1:5000` by default and automatically uses `PORT` when provided by the environment.

## Sightengine API Setup

Add these environment variables before running or deploying:

- `SIGHTENGINE_API_USER`
- `SIGHTENGINE_API_SECRET`

The image detection request is sent to:

- `https://api.sightengine.com/1.0/check.json`

Request configuration used:

- `models=genai`
- `media=image file upload`

## Deployment on Render

1. Push the project to GitHub under the `Albert-006` account.
2. Create a new Render Web Service.
3. Connect the GitHub repository.
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Add environment variables in Render:
   - `SECRET_KEY`
   - `SIGHTENGINE_API_USER`
   - `SIGHTENGINE_API_SECRET`
6. Upload or provision `model.pkl` and `vectorizer.pkl` in the deployed environment.
7. If you want SQLite history to persist across deploys, attach a persistent disk and keep the database file on that disk.

The included `Procfile` is:

```text
web: gunicorn app:app
```

## Security Notes

- No API keys are hardcoded
- Session cookies are configured as HTTP-only

## Troubleshooting

- If text analysis fails, verify `models/model.pkl` and `models/vectorizer.pkl` are present.
- If stopwords are missing, the app downloads the NLTK `stopwords` corpus once and reuses it afterward.
- If image analysis fails, verify your Sightengine credentials and network access.
- If uploads fail, ensure the image is PNG, JPG, JPEG, or WEBP and under 5 MB.

## Production Notes

- For real production use, replace SQLite with a managed database if you need multi-instance persistence.
- Set a strong `SECRET_KEY` in every deployed environment.
- Keep model artifacts out of public repositories if they are proprietary.
