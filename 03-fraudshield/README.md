# FraudShield — Fraud Classification Starter

Train and evaluate a classifier on reproducible synthetic transaction data. Demonstrates class imbalance, precision/recall, confusion matrix, and threshold-based review. This is not a real fraud detection service.

## Run

    pip install -r requirements.txt
    python train.py

The script prints evaluation metrics and saves a model to `artifacts/`. Real use requires vetted data, leakage checks, calibrated thresholds, security review, and human oversight.
