from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib


model = joblib("misinfo_model.pkl")


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = model.predict([text])[0]
    return jsonify({
        "verdict": prediction,
        "originalText": text
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)