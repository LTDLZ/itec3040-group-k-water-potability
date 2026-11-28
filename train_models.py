
import pandas as pd
import joblib
from pathlib import Path

from DataPreprocess import (
    preprocess_data,
    logistic_preprocessing,
    forest_preprocessing
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


# 1. Load preprocessed data (70/15/15 split)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()


# 2. Define classification models

# Majority-Class Baseline
baseline = DummyClassifier(strategy="most_frequent")

# Logistic Regression
logistic_model = Pipeline([
    *logistic_preprocessing().steps,
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Random Forest
random_forest_model = Pipeline([
    *forest_preprocessing().steps,
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

models = {
    "Majority-Class Baseline": baseline,
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}


# 3. Train classification models

print("\n--- Model Training ---")

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "trained successfully.")


# 4. Validation accuracy

print("\n--- Validation Results ---")

validation_scores = {}

for name, model in models.items():
    validation_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, validation_predictions)
    validation_scores[name] = accuracy

    print(f"{name}: {accuracy:.2%}")


# 5. Five-fold cross-validation

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n--- Cross-Validation Results ---")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(f"{name}: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")


# 6. Select model using validation accuracy

best_model = max(validation_scores, key=validation_scores.get)

print("\nBest model by validation accuracy:", best_model)


# 7. Final testing

print("\n--- Final Testing Results ---")

test_predictions = {}
test_probabilities = {}

for name, model in models.items():

    test_predictions[name] = model.predict(X_test)
    test_probabilities[name] = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        test_predictions[name]
    )

    print(f"{name}: {accuracy:.2%}")


# 8. Export trained models

Path("models").mkdir(exist_ok=True)

joblib.dump(baseline, "models/baseline.joblib")

joblib.dump(
    logistic_model,
    "models/logistic_regression.joblib"
)

joblib.dump(
    random_forest_model,
    "models/random_forest.joblib"
)

print("\nAll models saved successfully.")


# 9. Export predictions and probabilities

Path("results").mkdir(exist_ok=True)

predictions = pd.DataFrame({
    "sample_index": X_test.index,
    "Actual": y_test.to_numpy(),
    "Baseline": test_predictions["Majority-Class Baseline"],
    "Baseline_Probability": test_probabilities["Majority-Class Baseline"],
    "Logistic_Regression": test_predictions["Logistic Regression"],
    "Logistic_Probability": test_probabilities["Logistic Regression"],
    "Random_Forest": test_predictions["Random Forest"],
    "Random_Forest_Probability": test_probabilities["Random Forest"]
})

predictions.to_csv(
    "results/predictions.csv",
    index=False
)

print("\nAll predictions saved successfully.")
