# Predictive Lead Scoring for Martech

An end-to-end, reproducible scikit-learn example for scoring synthetic B2B marketing leads. The data is intentionally imbalanced, so model selection is assessed with precision, recall, F1-score, and a confusion matrix rather than accuracy alone.

## Quick start

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
```

The generator writes `leads_data.csv` (5,000 rows, approximately 15% conversions). Training uses an 80/20 stratified split and saves the complete preprocessing and Random Forest pipeline as `lead_scoring_pipeline.pkl`, plus `confusion_matrix.png` and `feature_importance_top5.png` under `artifacts/`.

## Features and preprocessing

Inputs include website time, email open rate, webinar attendance, company size, industry, and lead source. Numeric fields are median-imputed and standardized. Categorical fields are most-frequent-imputed and one-hot encoded with the first category dropped. The Random Forest uses `class_weight="balanced"` to account for class imbalance.

## Evaluation and business tradeoffs

The training script prints precision, recall, and F1 by class for held-out test leads. False positives spend sales time on leads that do not convert; false negatives leave potential customers untouched. The relative cost depends on the sales process, so choose a decision threshold based on observed campaign economics before deployment.

The feature importance chart shows the model's impurity-based importance over transformed columns. It describes model reliance, not causal influence. The generated data is synthetic and the model is an educational example, not validated for production decisions.
