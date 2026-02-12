TruthGuard – Fake News Detection System

TruthGuard is a desktop-based Fake News Detection application built using Machine Learning and Natural Language Processing (NLP). The system analyzes news text entered by the user and predicts whether it is Real or Fake along with a confidence score.

Technologies Used

Python

Tkinter

Scikit-learn

NLTK

TF-IDF Vectorizer

How It Works

The user enters news text into the application.

The text is preprocessed (cleaning, stopword removal, stemming).

TF-IDF converts the text into numerical features.

The trained machine learning model predicts whether the news is Real or Fake.

The result is displayed with a confidence percentage.

How to Run

Install required libraries and run:

python desk_app.py

Note

The model is based on statistical patterns in text data and may not fully understand context or factual accuracy. Predictions depend on the quality of the training dataset.
