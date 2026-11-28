"""Loads the team's saved models and predictions for the Water Potability dashboard.

Nothing is trained here: the models in models/*.joblib and the test-set predictions in
results/predictions.csv were produced by scripts/train_models.py (80/20 stratified split,
random_state=42). This module only loads them, rebuilds the same split, and computes metrics.

Run `python model.py` for a quick consistency check without the UI.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_DIR = ROOT / "models"
PRED_PATH = ROOT / "results" / "predictions.csv"

TARGET = "Potability"
RANDOM_STATE = 42  # same as scripts/train_models.py

# display name -> (model file, predicted-class column, probability column in predictions.csv)
MODEL_FILES = {
    "Majority-Class Baseline": ("baseline.joblib", "Baseline", "Baseline_Probability"),
    "Logistic Regression": ("logistic_regression.joblib", "Logistic_Regression", "Logistic_Probability"),
    "Random Forest": ("random_forest.joblib", "Random_Forest", "Random_Forest_Probability"),
}

FEATURE_INFO = {
    "ph": ("pH", "", "Acidity/alkalinity of water (WHO suggests 6.5-8.5)."),
    "Hardness": ("Hardness", "mg/L", "Capacity of water to precipitate soap (calcium/magnesium salts)."),
    "Solids": ("Total dissolved solids", "ppm", "Dissolved minerals and salts."),
    "Chloramines": ("Chloramines", "ppm", "Disinfectant; up to 4 ppm is considered safe."),
    "Sulfate": ("Sulfate", "mg/L", "Naturally occurring dissolved substance."),
    "Conductivity": ("Conductivity", "uS/cm", "Ability of water to conduct electricity (ionic content)."),
    "Organic_carbon": ("Organic carbon", "ppm", "Total organic carbon from decaying matter."),
    "Trihalomethanes": ("Trihalomethanes", "ug/L", "Disinfection by-products; up to 80 ppm is considered safe."),
    "Turbidity": ("Turbidity", "NTU", "Cloudiness caused by suspended particles."),
}
FEATURES = list(FEATURE_INFO)


# ------------------------------------------------------------------ loading
def missing_files() -> list:
    needed = [DATA_PATH, PRED_PATH] + [MODEL_DIR / f for f, _, _ in MODEL_FILES.values()]
    return [str(p.relative_to(ROOT)) for p in needed if not p.exists()]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Same 80/20 stratified split as scripts/train_models.py -> X_train, X_test, y_train, y_test."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def load_models() -> dict:
    """Fitted scikit-learn estimators saved with joblib (joblib files are pickles: only load trusted ones)."""
    return {name: joblib.load(MODEL_DIR / f) for name, (f, _, _) in MODEL_FILES.items()}


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PRED_PATH)


# ------------------------------------------------------------------ metrics
def metrics_at(y_true, proba, threshold=0.5) -> dict:
    """Predict 'potable' when probability is strictly above the threshold.

    Strict '>' matches scikit-learn's predict() at 0.5 (ties go to class 0), so the default
    threshold reproduces the predicted-class columns in predictions.csv exactly.
    """
    pred = (np.asarray(proba) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def roc_points(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    return fpr, tpr


def _final_step(est):
    return est.named_steps["classifier"] if hasattr(est, "named_steps") else est


def feature_importance(est) -> pd.Series:
    """Impurity importance for the forest, |standardised coefficient| for logistic regression."""
    clf = _final_step(est)
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=FEATURES).sort_values(ascending=False)


def cv_accuracy(models: dict, X_train, y_train) -> pd.DataFrame:
    """5-fold stratified cross-validation on the training split only (as in scripts/train_models.py)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, est in models.items():
        s = cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy")
        rows[name] = {"CV accuracy (mean)": s.mean(), "CV accuracy (std)": s.std()}
    return pd.DataFrame(rows).T


def consistency_check(models: dict, X_test, pred: pd.DataFrame) -> dict:
    """Do predictions.csv, the saved models and the rebuilt split agree with each other?"""
    out = {"same test rows": list(pred["sample_index"]) == list(X_test.index)}
    for name, (_, ccol, pcol) in MODEL_FILES.items():
        proba = models[name].predict_proba(X_test)[:, 1]
        out[name] = bool(
            np.array_equal(models[name].predict(X_test), pred[ccol].to_numpy())
            and np.allclose(proba, pred[pcol].to_numpy(), atol=1e-9)
        )
    return out


if __name__ == "__main__":
    gone = missing_files()
    if gone:
        raise SystemExit(f"Missing files: {gone}")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    models, pred = load_models(), load_predictions()
    print("Consistency:", consistency_check(models, X_test, pred))
    table = {n: metrics_at(pred["Actual"], pred[p]) for n, (_, _, p) in MODEL_FILES.items()}
    print(pd.DataFrame(table).T.round(3).to_string())
