"""Optional helper: replace the bundled demo CSV with IBM's public Telco sample."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "telco_customer_churn_demo.csv"
URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv",
]
REQUIRED = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    last_error = None
    for url in URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ChurnScope-Educational-Project/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            from io import BytesIO
            frame = pd.read_csv(BytesIO(content))
            missing = REQUIRED.difference(frame.columns)
            if missing:
                raise ValueError("Downloaded file is missing required columns: " + ", ".join(sorted(missing)))
            if len(frame) < 1000:
                raise ValueError(f"Downloaded file has only {len(frame)} rows; expected the full sample.")
            OUTPUT.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            print(f"Success: downloaded {len(frame):,} rows to {OUTPUT}")
            print(f"SHA-256: {digest}")
            print("The app will now use the IBM source CSV instead of the generated demo data.")
            return 0
        except Exception as exc:  # try the next public endpoint
            last_error = exc
    print("Could not download IBM's source CSV. Your bundled offline demo dataset has not been changed.")
    print(f"Last error: {last_error}")
    print("You can still run the app, or upload a compatible Telco CSV in the app sidebar.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
