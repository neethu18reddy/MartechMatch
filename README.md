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

The generator writes `leads_data.csv`. Training saves the complete preprocessing and Random Forest pipeline as `lead_scoring_pipeline.pkl`, plus evaluation charts under `artifacts/`.

## Dataset

There are 5,000 generated leads with six input features and a binary `Converted` target. Conversion is approximately 15%. A fixed random seed makes the dataset reproducible.

## Modeling

The preprocessing pipeline median-imputes and scales numeric columns, and most-frequent imputes then one-hot encodes categorical columns (dropping the first category). A stratified 80/20 split preserves the class balance. The Random Forest uses `class_weight="balanced"`.

Review precision, recall, and F1 by class in the printed classification report. False positives spend sales time on leads that do not convert; false negatives leave potential customers untouched. The relative cost depends on the sales process, so choose a decision threshold based on observed campaign economics before deployment.

Feature importance is the Random Forest's impurity-based importance over the transformed columns. It describes model reliance, not causal influence.
