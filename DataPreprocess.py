
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# 1) Data Cleaning

# Load dataset
waterset = pd.read_csv("water_potability.csv")

# Remove duplicate rows
waterset = waterset.drop_duplicates()

# Remove rows with missing Potability
waterset = waterset.dropna(subset=["Potability"])

# Remove invalid Potability labels
waterset = waterset[waterset["Potability"].isin([0, 1])]

# Separate features and target
xcolumn = waterset.drop(columns="Potability")
yrow = waterset["Potability"]

# Split into 70% training and 30% remaining
x_train, x_temp, y_train, y_temp = train_test_split(
    xcolumn,
    yrow,
    test_size=0.30,
    random_state=42,
    stratify=yrow
)

# Split remaining data into 15% validation and 15% testing
x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Display dataset information
print("Dataset shape:", waterset.shape)
print("Training samples:", len(x_train))
print("Validation samples:", len(x_val))
print("Testing samples:", len(x_test))
print("Input features:", xcolumn.columns.tolist())



# 2) Create the prediction models

# Model 1: Majority-Class Baseline
baseline = DummyClassifier(strategy="most_frequent")


# Model 2: Logistic Regression
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# Model 3: Random Forest
random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# Store all models in one dictionary
models = {
    "Majority-Class Baseline": baseline,
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}



# 3) Training the models


for name, model in models.items():

    # Teach the model using training data
    model.fit(x_train, y_train)

    print(name, "trained successfully.")



# 4) Measuring the accuracy of the models 

print("\n--- Validation Results ---")

for name, model in models.items():

    # Predict Potability using validation data
    validation_predictions = model.predict(x_val)

    # Calculate validation accuracy
    validation_accuracy = accuracy_score(
        y_val,
        validation_predictions
    )

    print(f"{name}: {validation_accuracy:.2%}")


# 5) Cross-Validation


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n--- Cross-Validation Results ---")

for name, model in models.items():

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(f"{name}: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")



# 6) Final Test


print("\n--- Final Testing Results ---")

test_results = {}

for name, model in models.items():

    # Predict Potability using testing data
    test_predictions = model.predict(x_test)

    # Calculate final accuracy
    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_results[name] = test_predictions

    print(f"{name}: {test_accuracy:.2%}")



# 7) Saving the Models


Path("models").mkdir(exist_ok=True)

joblib.dump(
    baseline,
    "models/baseline.joblib"
)

joblib.dump(
    logistic_model,
    "models/logistic_regression.joblib"
)

joblib.dump(
    random_forest_model,
    "models/random_forest.joblib"
)

print("\nAll models saved successfully.")



# 8) Store the Predictions


Path("results").mkdir(exist_ok=True)

predictions = pd.DataFrame({
    "sample_index": x_test.index,
    "Actual": y_test.to_numpy(),
    "Baseline": test_results["Majority-Class Baseline"],
    "Logistic_Regression": test_results["Logistic Regression"],
    "Random_Forest": test_results["Random Forest"]
})

predictions.to_csv(
    "results/predictions.csv",
    index=False
)

print("\nAll predictions saved successfully.")
