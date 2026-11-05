from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_data(rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate an offline demo dataset with the same core schema as IBM Telco churn."""
    rng = np.random.default_rng(seed)
    n = int(rows)

    gender = rng.choice(["Female", "Male"], size=n)
    senior = rng.binomial(1, 0.16, size=n)
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

    tenure = np.clip(np.rint(rng.gamma(shape=2.15, scale=16.5, size=n)), 1, 72).astype(int)
    zero_indices = rng.choice(n, size=min(11, n), replace=False)
    tenure[zero_indices] = 0

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.43, 0.22])
    phone = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52]))

    def internet_addon(probability: float) -> np.ndarray:
        return np.where(
            internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n, p=[probability, 1 - probability]),
        )

    online_security = internet_addon(0.36)
    online_backup = internet_addon(0.43)
    device_protection = internet_addon(0.42)
    tech_support = internet_addon(0.34)
    streaming_tv = internet_addon(0.48)
    streaming_movies = internet_addon(0.47)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.24, 0.21])
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.60, 0.40])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n,
        p=[0.34, 0.18, 0.24, 0.24],
    )

    addon_yes_count = (
        (multiple_lines == "Yes").astype(int)
        + (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )
    monthly = (
        18
        + np.where(internet == "DSL", 24, np.where(internet == "Fiber optic", 46, 0))
        + (phone == "Yes") * 18
        + addon_yes_count * 4.2
        + rng.normal(0, 5.5, size=n)
    )
    monthly = np.clip(monthly, 18.0, 120.0).round(2)
    total = np.maximum(0, monthly * tenure + rng.normal(0, 38, size=n)).round(2)
    total[tenure == 0] = np.nan

    # A plausible educational churn signal: short tenure, month-to-month contracts,
    # electronic checks, and missing support/security are associated with higher risk.
    log_odds = (
        -2.55
        + 1.05 * (contract == "Month-to-month")
        - 0.70 * (contract == "Two year")
        + 0.62 * (tenure <= 12)
        + 0.34 * (internet == "Fiber optic")
        + 0.52 * (payment == "Electronic check")
        + 0.42 * (tech_support == "No")
        + 0.30 * (online_security == "No")
        + 0.012 * (monthly - 65)
        + 0.20 * (paperless == "Yes")
        + rng.normal(0, 0.32, size=n)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    churn = np.where(rng.random(n) < probability, "Yes", "No")

    data = pd.DataFrame({
        "customerID": [f"DEMO-{index:05d}" for index in range(1, n + 1)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return data
