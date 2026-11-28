
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Load, clean and split the dataset
def preprocess_data():

    waterset = pd.read_csv("water_potability.csv")

    # Remove duplicate rows
    waterset = waterset.drop_duplicates()

    # Remove missing and invalid target labels
    waterset = waterset.dropna(subset=["Potability"])
    waterset = waterset[waterset["Potability"].isin([0, 1])]

    # Separate features and target
    X = waterset.drop(columns="Potability")
    y = waterset["Potability"]

    # 70% training and 30% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # 15% validation and 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Dataset shape:", waterset.shape)
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))
    print("Input features:", X.columns.tolist())

    return X_train, X_val, X_test, y_train, y_val, y_test


# Preprocessing for Logistic Regression
def logistic_preprocessing():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


# Preprocessing for Random Forest
def forest_preprocessing():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
