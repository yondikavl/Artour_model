# app.py

from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# Load model dan vectorizer dari file pkl
bundle = joblib.load("spam_model_bundle.pkl")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:8000", "http://127.0.0.1:5173"])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    # Transformasi dan prediksi
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]

    vec  = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])          # 0 / 1
    return jsonify({"prediction": pred})       # ← field bernama prediction
                                               #     (biar konsisten di Nest)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
