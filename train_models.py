import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("water_potability.csv")

# Define features and target
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset shape:", df.shape)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X.columns.tolist())


# Model 1: Majority-Class Baseline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

baseline = DummyClassifier(strategy="most_frequent")

# Train the baseline model
baseline.fit(X_train, y_train)

# Generate predictions
baseline_predictions = baseline.predict(X_test)

# Calculate accuracy
baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

print("\n--- Majority-Class Baseline ---")
print(f"Accuracy: {baseline_accuracy:.2%}")

# Model 2: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\n--- Logistic Regression ---")
print(f"Accuracy: {logistic_accuracy:.2%}")

# Model 3: Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)
random_forest_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

random_forest_accuracy = accuracy_score(
    y_test, random_forest_predictions
)

print("\n--- Random Forest ---")
print(f"Accuracy: {random_forest_accuracy:.2%}")

# Cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Majority-Class Baseline": baseline,
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

print("\n--- Cross-Validation Results ---")

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(f"{name}: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")

    # Export trained models
import joblib
from pathlib import Path

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

# Export model predictions
Path("results").mkdir(exist_ok=True)

predictions = pd.DataFrame({
    "sample_index": X_test.index,
    "Actual": y_test.to_numpy(),
    "Baseline": baseline_predictions,
    "Baseline_Probability": baseline.predict_proba(X_test)[:, 1],
    "Logistic_Regression": logistic_predictions,
    "Logistic_Probability": logistic_probabilities,
    "Random_Forest": random_forest_predictions,
    "Random_Forest_Probability": random_forest_probabilities
})

predictions.to_csv("results/predictions.csv", index=False)

print("\nAll predictions saved successfully.")