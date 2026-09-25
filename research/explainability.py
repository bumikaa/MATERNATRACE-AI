import os
import joblib
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# ============================================================
# GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

if hasattr(classifier, "coef_"):

    importance_values = abs(
        classifier.coef_[0]
    )

elif hasattr(classifier, "feature_importances_"):

    importance_values = classifier.feature_importances_

else:

    raise ValueError(
        "This model does not provide built-in feature importance."
    )


# ============================================================
# CREATE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 60)
print("MATERNATRACE AI FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.head(15).to_string(
        index=False
    )
)


print("\nExplainability analysis complete.")