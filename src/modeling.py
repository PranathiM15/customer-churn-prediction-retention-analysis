from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .data_processing import model_features


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])


def train_model_bundle(data: pd.DataFrame, random_state: int = 42) -> dict:
    """Train and compare three classifiers using a stratified holdout set."""
    X, y = model_features(data)
    if y.nunique() < 2:
        raise ValueError("The Churn column must contain both Yes and No customers to train models.")
    if len(data) < 100:
        raise ValueError("Please use at least 100 customer records for a meaningful model split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1200, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=18, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        ),
    }

    results: dict[str, dict] = {}
    metric_rows = []
    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        report = classification_report(
            y_test, predictions, labels=[0, 1], target_names=["Retained", "Churned"],
            output_dict=True, zero_division=0
        )
        results[name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
            "classification_report": report,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "metrics": metrics,
        }
        metric_rows.append(metrics)

    metrics_frame = pd.DataFrame(metric_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = str(metrics_frame.iloc[0]["Model"])
    best_pipeline = results[best_name]["pipeline"]

    # Model-agnostic importance: how much each original field changes ROC-AUC when shuffled.
    importance = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=2,
        random_state=random_state,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance.importances_mean,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "results": results,
        "metrics": metrics_frame,
        "best_model": best_name,
        "feature_importance": importance_frame,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "test_churn_rate": float(y_test.mean()),
        "random_state": random_state,
    }
