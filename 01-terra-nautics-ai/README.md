# TerraNautics AI — Real Estate Intelligence

A small, reproducible starter app for exploring real-estate price estimation. It trains a baseline regression model on synthetic data so the workflow runs without downloading private datasets.

## Features
- Generate a reproducible demo dataset.
- Train a scikit-learn regression model and report MAE/RMSE/R².
- Estimate a price from area, bedrooms, age, and location score.
- Streamlit UI for interactive experimentation.

## Run

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

This is an educational demo, not a property valuation or financial-advice service. Replace synthetic data with licensed, verified local data before drawing real conclusions. Azure ML deployment can be added using your own Azure subscription.
