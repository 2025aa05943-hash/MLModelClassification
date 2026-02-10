# BINARY CLASSIFICATION PROBLEM- BREAST CANCER DETECTION

import numpy as np
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
#Naive Bayes classifier- Gaussian
from sklearn.naive_bayes import GaussianNB
#random forest classifier
from sklearn.ensemble import RandomForestClassifier
#XGBoost
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(X.shape)

#Split Train and Test Data


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
}

# Fit the training data in above models

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained successfully")

# Evaluate the models

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    # Get probabilities or decision scores for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# -------------------------------
# Results comparison table
# -------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print(results_df)

#Save Each Trained Model to Pickle
# directory to save models
os.makedirs("model/saved_models", exist_ok=True)

for name, model in models.items():
    filename = f"model/saved_models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"{name} saved to {filename}")

# Save test samples for app testing
test_samples = X_test.copy()
test_samples["target"] = y_test
test_samples.to_csv("model/saved_models/test_samples.csv", index=False)
print("Test samples saved to model/saved_models/test_samples.csv")

