# MLOps Sentinel — Model Quality and Drift Monitor

A lightweight MLOps starter that trains a regression model, stores evaluation metadata, and compares incoming feature distributions with a reference sample. Demonstrates reproducible artifacts and basic drift signals without requiring cloud credentials.

## Run

    pip install -r requirements.txt
    python monitor.py

This is a teaching example, not a production monitoring system. Extend it with MLflow tracking, CI tests, model registry approvals, Azure ML endpoints, and alerting before real deployment.
