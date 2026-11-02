from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.ml.churn_data import split_churn_data
from src.ml.churn_models import build_logistic_regression, build_random_forest


# Customers with a chance of leaving at or above this value are flagged.
THRESHOLD = 0.4

MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "churn_model.joblib"
)

X_train, X_test, y_train, y_test = split_churn_data()


def cv_score(model):
    """Average ROC-AUC across 5 folds of the training data."""
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )
    return scores.mean()


# Step 1: Tune the Random Forest
print("TUNING THE RANDOM FOREST (5-fold cross-validation on training data)")
print("max_depth | mean ROC-AUC")
print("----------|-------------")

best_depth, best_forest_score = None, 0.0

for depth in [2, 3, 4, 6, 8, 12]:
    score = cv_score(build_random_forest(depth))
    print(f"{depth:>9} | {score:.3f}")

    if score > best_forest_score:
        best_depth, best_forest_score = depth, score

print(f"Best max_depth: {best_depth}")
print()


# Step 2: Compare the models
candidates = {
    "Logistic regression": build_logistic_regression(),
    f"Random forest (max_depth={best_depth})": build_random_forest(best_depth),
}

print("COMPARING THE MODELS (mean ROC-AUC, same method)")

cv_results = {}

for name, model in candidates.items():
    cv_results[name] = cv_score(model)
    print(f"{name:<32} {cv_results[name]:.3f}")

winner_name = max(cv_results, key=cv_results.get)
winner = candidates[winner_name]

print(f"Winner: {winner_name}")
print()


# Step 3: Test the winning model once
winner.fit(X_train, y_train)

probabilities = winner.predict_proba(X_test)[:, 1]
flagged = (probabilities >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()

print(f"FINAL TEST RESULTS for {winner_name} at threshold {THRESHOLD}")
print(f"accuracy  {accuracy_score(y_test, flagged):.3f}")
print(f"precision {precision_score(y_test, flagged):.3f}")
print(f"recall    {recall_score(y_test, flagged):.3f}")
print(f"f1        {f1_score(y_test, flagged):.3f}")
print(f"roc_auc   {roc_auc_score(y_test, probabilities):.3f}")

print(f"Caught {tp} of {tp + fn} leavers, with {fp} false alarms")
print()


# Step 4: Save the winning model
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(
    {
        "model": winner,
        "model_name": winner_name,
        "features": list(X_train.columns),
        "threshold": THRESHOLD,
    },
    MODEL_PATH,
)

print(
    f"Saved model to {MODEL_PATH} "
    f"({MODEL_PATH.stat().st_size / 1024:.0f} KB)"
)


# Step 5: Reload the saved model and check it
saved = joblib.load(MODEL_PATH)

one_customer = X_test.iloc[[0]]
chance = saved["model"].predict_proba(one_customer)[0, 1]

decision = (
    "FLAG for retention offer"
    if chance >= saved["threshold"]
    else "no action"
)

print(
    f"Reload check: chance of leaving = {chance:.3f} -> {decision}"
)