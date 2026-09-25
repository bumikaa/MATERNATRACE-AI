import os
import joblib
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "birth_weight_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.joblib"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "age",
    "pre_pregnancy_bmi",
    "gestational_age_weeks",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "hemoglobin_level",
    "number_of_prenatal_visits",
    "has_diabetes",
    "has_hypertension",
    "smoking_status",
    "alcohol_consumption",
    "education_level",
    "household_income",
    "iron_supplementation"
]

TARGET = "birth_weight_category"


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

missing_columns = [
    column for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# 5. SPLIT DATA
# ============================================================

X = df[FEATURES]
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

NUMERICAL_FEATURES = [
    "age",
    "pre_pregnancy_bmi",
    "gestational_age_weeks",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "hemoglobin_level",
    "number_of_prenatal_visits"
]

CATEGORICAL_FEATURES = [
    "has_diabetes",
    "has_hypertension",
    "smoking_status",
    "alcohol_consumption",
    "education_level",
    "household_income",
    "iron_supplementation"
]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])


preprocessor = ColumnTransformer([
    (
        "numerical",
        numeric_pipeline,
        NUMERICAL_FEATURES
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    )
])


# ============================================================
# 7. MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# 8. TRAIN
# ============================================================

print("Training model...")

pipeline.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 9. EVALUATION
# ============================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Evaluation")
print("================")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"\nModel saved to:")
print(MODEL_PATH)


# ============================================================
# 11. FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# 12. HOME ROUTE
# ============================================================

@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "maternatrace_ai_public_platform.html"
    )

# ============================================================
# 13. PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400


        # Create dataframe from received input
        input_data = pd.DataFrame(
            [data],
            columns=FEATURES
        )


        # Make prediction
        prediction = pipeline.predict(
            input_data
        )[0]


        # Probability
        probabilities = pipeline.predict_proba(
            input_data
        )[0]


        classes = pipeline.classes_


        probability_dict = {
            str(cls): round(
                float(prob) * 100,
                2
            )
            for cls, prob in zip(
                classes,
                probabilities
            )
        }


        return jsonify({

            "success": True,

            "prediction": str(prediction),

            "probabilities": probability_dict,

            "message":
                "This is a machine-learning risk estimate "
                "and not a medical diagnosis."

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# ============================================================
# 14. RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("        MaternaTrace AI Backend")
    print("========================================")

    print("\nAPI running at:")
    print("http://127.0.0.1:5000")

    print("\nPrediction endpoint:")
    print("POST http://127.0.0.1:5000/predict")

    print("\nPress CTRL+C to stop the server.")

    app.run(host="127.0.0.1", port=5000, debug=False)