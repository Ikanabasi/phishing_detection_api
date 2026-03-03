from flask import Flask, request, jsonify
import pickle
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
with open("phishing_model.pkl", "rb") as f:
    model = pickle.load(f)

suspicious_words = ["login", "verify", "secure", "update", "bank", "account"]

def has_suspicious(url):
    return int(any(word in str(url).lower() for word in suspicious_words))

def extract_features(url):
    url = str(url)
    return [
        len(url),                               # url_length
        url.count("."),                         # num_dots
        1 if url.startswith("https") else 0,    # has_https
        1 if "@" in url else 0,                 # has_at
        1 if "-" in url else 0,                 # has_hyphen
        1 if re.search(r"\d", url) else 0,      # has_digits
        has_suspicious(url)                     # has_suspicious_word
    ]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data["url"]

    features = extract_features(url)
    prediction = int(model.predict([features])[0])

    return jsonify({"prediction": prediction})

@app.route("/")
def home():
    return "Phishing Detection API is running"

if __name__ == "__main__":
    app.run()
