"""Train, evaluate, and serialize the lead conversion classifier."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "leads_data.csv"
NUMERIC_FEATURES = ["Time_on_Website", "Email_Open_Rate", "Webinar_Attendance"]
CATEGORICAL_FEATURES = ["Company_Size", "Industry", "Lead_Source"]
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
    data = pd.read_csv(DATA_PATH)
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    print(f"Loaded {len(data):,} leads. Run the model pipeline to train and evaluate.")


if __name__ == "__main__":
    main()
