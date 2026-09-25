import os
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
HTML_PATH = os.path.join(BASE_DIR, "maternatrace_ai_public_platform.html")

app = Flask(__name__)
CORS(app)

print("Loading MaternaTrace AI model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("MaternaTrace AI backend is running.")


@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "maternatrace_ai_public_platform.html"
    )


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "project": "MaternaTrace AI",
        "status": "running",
        "model": "loaded"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No input data received"
        }), 400

    try:
        prediction = model.predict([data])[0]

        probabilities = model.predict_proba([data])[0]

        classes = model.classes_

        probability_dict = {
            str(classes[i]): round(float(probabilities[i]), 4)
            for i in range(len(classes))
        }

        return jsonify({
            "prediction": str(prediction),
            "probabilities": probability_dict,
            "message": "This is a machine-learning risk estimate, not a medical diagnosis."
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )