from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

OUT = Path(__file__).parent / 'artifacts'
FEATURES = ['temperature', 'demand_index', 'promotion']


def main():
    rng = np.random.default_rng(10)
    n = 2000
    reference = pd.DataFrame({'temperature': rng.normal(22, 6, n), 'demand_index': rng.normal(50, 12, n), 'promotion': rng.integers(0, 2, n)})
    target = 3.5 * reference.temperature + 1.8 * reference.demand_index + 12 * reference.promotion + rng.normal(0, 10, n)
    X_train, X_test, y_train, y_test = train_test_split(reference, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=140, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {'mae': float(mean_absolute_error(y_test, pred)), 'rmse': float(mean_squared_error(y_test, pred) ** 0.5), 'r2': float(r2_score(y_test, pred))}
    # Simulate a shifted batch to demonstrate a simple standardized mean-shift signal.
    incoming = reference.sample(400, random_state=2).copy()
    incoming['temperature'] += 8
    drift = {}
    for col in FEATURES:
        ref_std = float(reference[col].std()) or 1.0
        shift = abs(float(incoming[col].mean() - reference[col].mean())) / ref_std
        drift[col] = {'standardized_mean_shift': round(shift, 3), 'alert': bool(shift >= 0.5)}
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / 'model.joblib')
    (OUT / 'report.json').write_text(json.dumps({'metrics': metrics, 'drift': drift}, indent=2), encoding='utf-8')
    print('Holdout metrics:', json.dumps(metrics, indent=2))
    print('Feature drift report:', json.dumps(drift, indent=2))
    print('Artifacts saved under', OUT)

if __name__ == '__main__':
    main()
