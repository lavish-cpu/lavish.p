from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / 'artifacts'


def main():
    X, y = make_classification(n_samples=12000, n_features=12, n_informative=7, n_redundant=3,
        weights=[0.985, 0.015], class_sep=1.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=180, class_weight='balanced', random_state=42, n_jobs=-1, min_samples_leaf=2))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = 0.35
    predictions = (probabilities >= threshold).astype(int)
    print('Threshold:', threshold)
    print('Average precision:', round(average_precision_score(y_test, probabilities), 4))
    print('ROC-AUC:', round(roc_auc_score(y_test, probabilities), 4))
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / 'fraud_model.joblib')
    print('Saved model to', OUT / 'fraud_model.joblib')

if __name__ == '__main__':
    main()
