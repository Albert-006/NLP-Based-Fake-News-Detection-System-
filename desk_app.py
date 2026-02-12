import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

def predict_news():
    news_text = text_box.get("1.0", tk.END)
    
    if news_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter news text.")
        return
    
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    confidence = model.predict_proba(vector).max() * 100
    
    if prediction[0] == 1:
        result_label.config(text=f"Fake News ({confidence:.2f}% confidence)", fg="red")
    else:
        result_label.config(text=f"Real News ({confidence:.2f}% confidence)", fg="green")

# Create window
root = tk.Tk()
root.title("Fake News Detection System")
root.geometry("700x500")
root.resizable(False, False)

# Title
title_label = tk.Label(root, text="Fake News Detection System", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Text box
text_box = tk.Text(root, height=15, width=80)
text_box.pack(pady=10)

# Button
check_button = tk.Button(root, text="Check News", font=("Arial", 12), command=predict_news)
check_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

# Run app
root.mainloop()
