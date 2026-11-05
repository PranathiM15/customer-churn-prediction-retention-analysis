from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "ServiceCount"]
TARGET_COLUMN = "ChurnFlag"


def clean_telco_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a Telco-style customer churn dataframe."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The uploaded dataset is empty or could not be read.")

    df = data.copy()
    df.columns = [str(column).strip() for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "This file does not look like the Telco Customer Churn dataset. "
            "Missing columns: " + ", ".join(missing)
        )

    # Remove exact duplicate customer rows before preparing the target.
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"], keep="first")
    else:
        df = df.drop_duplicates()

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).clip(lower=0, upper=72)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    monthly_median = float(df["MonthlyCharges"].median()) if df["MonthlyCharges"].notna().any() else 0.0
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(monthly_median)

    # In the original IBM sample, a small number of new customers have blank TotalCharges.
    zero_tenure = df["tenure"].eq(0)
    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    total_median = float(df["TotalCharges"].median()) if df["TotalCharges"].notna().any() else 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(total_median).clip(lower=0)

    churn_text = df["Churn"].astype(str).str.strip().str.lower()
    target_map = {"yes": "Yes", "no": "No", "1": "Yes", "0": "No", "true": "Yes", "false": "No"}
    mapped = churn_text.map(target_map)
    if mapped.isna().any():
        bad_values = sorted(churn_text[mapped.isna()].dropna().unique().tolist())
        raise ValueError("Churn must contain Yes/No values. Unexpected values: " + ", ".join(bad_values[:8]))
    df["Churn"] = mapped
    df[TARGET_COLUMN] = df["Churn"].eq("Yes").astype(int)

    # Keep a stable representation for categoricals and prevent blank strings from becoming categories.
    categorical_columns = [column for column in REQUIRED_COLUMNS if column not in {
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
    }]
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df.loc[df[column].eq(""), column] = "Unknown"

    return add_engineered_features(df)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add tenure bands and a simple count of subscribed services."""
    df = data.copy()
    tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["TenureBand"] = pd.cut(
        tenure,
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
        include_lowest=True,
    ).astype(str)

    service_columns = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    flags = []
    for column in service_columns:
        if column in df.columns:
            flags.append(df[column].astype(str).str.lower().eq("yes").astype(int))
    if "InternetService" in df.columns:
        flags.append(df["InternetService"].astype(str).str.lower().ne("no").astype(int))
    df["ServiceCount"] = sum(flags) if flags else 0
    return df


def model_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the binary churn target."""
    excluded = {"customerID", "Churn", TARGET_COLUMN}
    feature_columns = [column for column in data.columns if column not in excluded]
    X = data[feature_columns].copy()
    y = data[TARGET_COLUMN].astype(int).copy()
    return X, y


def make_customer_record(values: dict) -> pd.DataFrame:
    """Create a one-row model input from the risk simulator form."""
    row = dict(values)
    row["TotalCharges"] = float(row.get("MonthlyCharges", 0)) * float(row.get("tenure", 0))
    frame = pd.DataFrame([row])
    return add_engineered_features(frame)


def segment_churn_rates(data: pd.DataFrame, segment: str) -> pd.DataFrame:
    if segment not in data.columns:
        return pd.DataFrame(columns=[segment, "Customers", "Churn rate"])
    result = (
        data.groupby(segment, dropna=False)
        .agg(Customers=(TARGET_COLUMN, "size"), **{"Churn rate": (TARGET_COLUMN, "mean")})
        .reset_index()
    )
    result["Churn rate"] = result["Churn rate"].astype(float)
    return result.sort_values("Churn rate", ascending=False)
