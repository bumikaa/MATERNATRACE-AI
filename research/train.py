import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "birth_weight_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# TARGET
# ============================================================

TARGET = "birth_weight_category"

df = df.dropna(subset=[TARGET])

y = df[TARGET].map({
    "Normal": 0,
    "Low": 1
})

X = df.drop(columns=[TARGET])


# ============================================================
# FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "age",
    "pre_pregnancy_bmi",
    "gestational_age_weeks",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "hemoglobin_level",
    "number_of_prenatal_visits",
    "household_income"
]

CATEGORICAL_FEATURES = [
    "smoking_status",
    "alcohol_consumption",
    "education_level",
    "has_diabetes",
    "has_hypertension",
    "iron_supplementation"
]


# ============================================================
# PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        random_state=42,
        solver="liblinear",
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=(
            (y_train == 0).sum() /
            (y_train == 1).sum()
        )
    )
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


results = {}


print("\nRunning 5-Fold Stratified Cross-Validation...")
print("=" * 60)


for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results[name] = {
        metric: np.mean(scores[f"test_{metric}"])
        for metric in scoring
    }

    print(f"\n{name}")

    for metric, value in results[name].items():
        print(f"{metric.upper():10}: {value:.4f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

print("\n\nMODEL COMPARISON")
print("=" * 60)
print(results_df.round(4))


# ============================================================
# SELECT BEST MODEL USING F1
# ============================================================

best_model_name = results_df["f1"].idxmax()

print(
    f"\nBest model based on cross-validation F1: "
    f"{best_model_name}"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

final_classifier = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", final_classifier)
])

print("\nTraining final model...")

final_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    final_pipeline,
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)

print("\nTraining complete.")