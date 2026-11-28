
# ITEC 3040 - Group K
# Model Evaluation and Interpretation
# Evaluates the predictions from the three classification models.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# 1. Load the predictions from Omar's model training.
project_folder = Path(__file__).resolve().parent
predictions_file = project_folder / "results" / "predictions.csv"

data = pd.read_csv(predictions_file)
y_actual = data["Actual"]

# Potability = 1 means potable water.
# Potability = 0 means non-potable water.

models = {
    "Majority-Class Baseline": (
        "Baseline",
        "Baseline_Probability"
    ),
    "Logistic Regression": (
        "Logistic_Regression",
        "Logistic_Probability"
    ),
    "Random Forest": (
        "Random_Forest",
        "Random_Forest_Probability"
    )
}

# Create a folder for our evaluation results.
output_folder = project_folder / "results" / "evaluation"
output_folder.mkdir(parents=True, exist_ok=True)

evaluation_results = []

# 2. Calculate the evaluation metrics.
for name, (prediction_col, probability_col) in models.items():

    y_pred = data[prediction_col]
    y_prob = data[probability_col]

    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(
        y_actual, y_pred, zero_division=0
    )
    recall = recall_score(
        y_actual, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_actual, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_actual, y_prob)

    evaluation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

    # 3. Create a confusion matrix for each model.
    cm = confusion_matrix(
        y_actual, y_pred, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-potable", "Potable"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(
        output_folder / f"confusion_matrix_{filename}.png",
        dpi=300
    )
    plt.close()

# 4. Save and print the model comparison table.
results_df = pd.DataFrame(evaluation_results)

results_df.to_csv(
    output_folder / "evaluation_metrics.csv",
    index=False
)

print("\nMODEL EVALUATION RESULTS")
print("-" * 75)

print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

# 5. Generate ROC curves for all three models.
fig, ax = plt.subplots(figsize=(8, 6))

for name, (prediction_col, probability_col) in models.items():

    RocCurveDisplay.from_predictions(
        y_actual,
        data[probability_col],
        name=name,
        ax=ax
    )

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Random classification"
)

ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(
    output_folder / "roc_curve_comparison.png",
    dpi=300
)
plt.close()

print("\nEvaluation completed successfully.")
print("Results saved in: results/evaluation/")


# 6. Random Forest Feature Importance
# Check which water-quality measurements the model used most.

import joblib

random_forest = joblib.load(
    project_folder / "models" / "random_forest.joblib"
)

# Get the trained classifier from the saved pipeline.
classifier = random_forest.named_steps["classifier"]

# Get the feature names and importance values.
feature_names = random_forest.feature_names_in_
importance_values = classifier.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort features from most important to least important.
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Save the feature-importance results.
feature_importance.to_csv(
    output_folder / "feature_importance.csv",
    index=False
)

# Create the feature-importance graph.
plt.figure(figsize=(9, 6))
plt.barh(
    feature_importance["Feature"][::-1],
    feature_importance["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Water Quality Measurement")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    output_folder / "random_forest_feature_importance.png",
    dpi=300
)

plt.close()

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(feature_importance.to_string(index=False))
print("\nFeature importance graph saved successfully.")
