import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "birth_weight_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

TARGET = "birth_weight_category"

df = df.dropna(subset=[TARGET])


# ============================================================
# TARGET ENCODING
# ============================================================

y = df[TARGET].map({
    "Normal": 0,
    "Low": 1
})

X = df.drop(columns=[TARGET])


# Safety check
if y.isna().any():
    raise ValueError(
        "Target contains values other than 'Normal' and 'Low'."
    )


# ============================================================
# SAME TEST SPLIT USED DURING TRAINING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)


# ============================================================
# MODEL CLASSES
# ============================================================

print("\nModel classes:")

if hasattr(model, "classes_"):
    print(model.classes_)
else:
    raise AttributeError(
        "The loaded model does not expose classes_."
    )


# ============================================================
# PREDICTION
# ============================================================

y_pred_raw = model.predict(X_test)


# Convert model output to the same numeric labels used by y_test
if y_pred_raw.dtype.kind in "OUS":
    y_pred = pd.Series(y_pred_raw).map({
        "Normal": 0,
        "Low": 1
    }).values
else:
    y_pred = y_pred_raw.astype(int)


# Safety check
if pd.isna(y_pred).any():
    raise ValueError(
        "Model produced an unexpected class label."
    )


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

if not hasattr(model, "predict_proba"):
    raise AttributeError(
        "The loaded model does not support predict_proba()."
    )

probabilities = model.predict_proba(X_test)

classes = list(model.classes_)


# Find probability of the LOW class explicitly
if "Low" in classes:

    low_index = classes.index("Low")

    y_probability = probabilities[:, low_index]

else:
    raise ValueError(
        f"'Low' class not found in model classes: {classes}"
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ROC-AUC uses probability of the positive class = Low
roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MATERNATRACE AI MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nPositive class for Precision / Recall / F1 / ROC-AUC: Low")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Low"],
        zero_division=0
    )
)


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

print("Test set size:", len(y_test))

print(
    "Actual class distribution:"
)

print(
    pd.Series(y_test)
    .map({
        0: "Normal",
        1: "Low"
    })
    .value_counts()
)


print("\nEvaluation complete.")