"""Train and export a reproducible B2B lead-conversion pipeline."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_LEADS = 5_000
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "leads_data.csv"
MODEL_PATH = ROOT / "martech_pipeline.pkl"
IMPORTANCE_PATH = ROOT / "feature_importance.png"

NUMERIC_FEATURES = [
    "Time_on_Website",
    "Email_Open_Rate",
    "Webinar_Attendance",
]
CATEGORICAL_FEATURES = ["Industry", "Company_Size"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Converted"
INDUSTRIES = ["Tech", "Healthcare", "Finance", "Retail"]
COMPANY_SIZES = ["Startup", "Mid-Market", "Enterprise"]


def generate_synthetic_data(n_rows: int = N_LEADS, seed: int = SEED) -> pd.DataFrame:
    """Create synthetic leads with a learnable signal and exactly 15% positives."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "Time_on_Website": np.clip(
                rng.gamma(shape=2.2, scale=110.0, size=n_rows), 5.0, 1_800.0
            ),
            "Email_Open_Rate": rng.beta(a=2.2, b=3.0, size=n_rows),
            "Webinar_Attendance": rng.binomial(1, 0.28, size=n_rows),
            "Industry": rng.choice(
                INDUSTRIES, size=n_rows, p=[0.35, 0.25, 0.22, 0.18]
            ),
            "Company_Size": rng.choice(
                COMPANY_SIZES, size=n_rows, p=[0.42, 0.38, 0.20]
            ),
        }
    )

    # A noisy latent propensity creates realistic overlap while retaining a
    # meaningful relationship between engagement, firmographics, and conversion.
    website_signal = (data["Time_on_Website"] - data["Time_on_Website"].mean()) / data[
        "Time_on_Website"
    ].std()
    email_signal = (data["Email_Open_Rate"] - data["Email_Open_Rate"].mean()) / data[
        "Email_Open_Rate"
    ].std()
    score = (
        0.9 * website_signal
        + 1.05 * email_signal
        + 0.85 * data["Webinar_Attendance"]
        + 0.65 * (data["Company_Size"] == "Enterprise").astype(float)
        + 0.25 * (data["Company_Size"] == "Mid-Market").astype(float)
        + 0.18 * (data["Industry"] == "Tech").astype(float)
        + 0.08 * (data["Industry"] == "Healthcare").astype(float)
        + rng.normal(0.0, 0.25, size=n_rows)
    )

    # Select the top 15% of latent propensity scores for an exactly stratifiable,
    # reproducible class balance (750 positive examples for the default row count).
    positive_count = max(1, int(round(n_rows * 0.15)))
    positive_indices = np.argpartition(score.to_numpy(), -positive_count)[-positive_count:]
    data[TARGET_COLUMN] = 0
    data.loc[data.index[positive_indices], TARGET_COLUMN] = 1
    return data


def build_pipeline() -> Pipeline:
    """Build preprocessing and classifier as one inference-safe object."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def save_feature_importance(pipeline: Pipeline, output_path: Path) -> pd.DataFrame:
    """Save a chart and return importances for the fitted transformed features."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    importance_table = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    chart_data = importance_table.head(10).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(chart_data["Feature"], chart_data["Importance"], color="#3977F6")
    ax.set_title("Lead Conversion | Feature Importance", loc="left", weight="bold")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_ylabel("")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return importance_table


def main() -> None:
    data = generate_synthetic_data()
    data.to_csv(DATA_PATH, index=False)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print(f"Dataset: {len(data):,} leads | converted: {y.mean():.1%}")
    print(f"Train/test rows: {len(X_train):,}/{len(X_test):,} (stratified)")
    print("\nClassification report (held-out test set):")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not converted", "Converted"],
            digits=3,
            zero_division=0,
        )
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=[1], zero_division=0
    )
    print(
        "Converted leads — "
        f"precision: {precision[0]:.3f} | recall: {recall[0]:.3f} | "
        f"F1-score: {f1[0]:.3f}"
    )

    # Refit on all labeled rows for the deployed artifact after recording the
    # untouched holdout metrics. The test set is not used to choose parameters.
    pipeline.fit(X, y)
    importance_table = save_feature_importance(pipeline, IMPORTANCE_PATH)
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    print(f"\nSaved dataset: {DATA_PATH}")
    print(f"Saved feature importance chart: {IMPORTANCE_PATH}")
    print(f"Saved fitted pipeline: {MODEL_PATH}")
    print("Top five features:")
    print(importance_table.head(5).to_string(index=False, formatters={"Importance": "{:.4f}".format}))


if __name__ == "__main__":
    main()

