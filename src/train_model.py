"""Train, evaluate, and serialize the lead conversion classifier."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "leads_data.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
NUMERIC_FEATURES = ["Time_on_Website", "Email_Open_Rate", "Webinar_Attendance"]
CATEGORICAL_FEATURES = ["Company_Size", "Industry", "Lead_Source"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Converted"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("Classification report (held-out test set):")
    print(classification_report(y_test, predictions, target_names=["Not converted", "Converted"], zero_division=0))

    # False positives consume sales capacity on leads that will not convert.
    # False negatives miss potential customers and can sacrifice pipeline revenue.
    # The confusion matrix helps teams weigh these costs against their sales capacity.
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not converted", "Converted"],
        yticklabels=["Not converted", "Converted"],
        ax=ax,
    )
    ax.set(title="Lead Conversion Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix to {ARTIFACTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
