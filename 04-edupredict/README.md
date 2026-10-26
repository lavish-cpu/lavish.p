# EduPredict — Student Performance Analytics

A transparent educational ML example that estimates a numeric assessment score from attendance, study hours, and prior score using synthetic data. Intended to demonstrate regression and responsible use, not to make real decisions about students.

## Run

    pip install -r requirements.txt
    python train.py

The script reports holdout metrics and feature importance. Do not use predictions to label, punish, or deny opportunities to students. Any real deployment needs consent, privacy safeguards, bias analysis, and educator review.
