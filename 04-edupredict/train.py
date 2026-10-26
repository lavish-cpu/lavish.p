import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def make_data(n=1500, seed=21):
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(45, 100, n)
    study_hours = rng.uniform(0, 6, n)
    previous_score = rng.uniform(25, 100, n)
    score = 0.38 * previous_score + 0.32 * attendance + 4.2 * study_hours + rng.normal(0, 8, n)
    score = np.clip(score, 0, 100)
    return pd.DataFrame({'attendance_pct': attendance, 'study_hours_daily': study_hours, 'previous_score': previous_score, 'assessment_score': score})


def main():
    df = make_data()
    X = df.drop(columns='assessment_score')
    y = df['assessment_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f'MAE: {mean_absolute_error(y_test, pred):.2f} points')
    print(f'RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f} points')
    print(f'R²: {r2_score(y_test, pred):.3f}')
    print('\nFeature importance:')
    print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).to_string())
    print('\nSynthetic data only; do not use as the basis for real student decisions.')

if __name__ == '__main__':
    main()
