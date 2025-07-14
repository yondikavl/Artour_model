from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000", "http://127.0.0.1:5173"])

def load_model(path="spam_model_filter.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["vectorizer"]

model, vectorizer = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    vectorized_text = vectorizer.transform([text])
    prediction = int(model.predict(vectorized_text)[0])

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


