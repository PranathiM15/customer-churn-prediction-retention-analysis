from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import clean_telco_dataframe, make_customer_record, model_features
from src.modeling import train_model_bundle
from src.sample_data import generate_sample_data


class ChurnProjectTests(unittest.TestCase):
    def test_demo_dataset_has_expected_shape_and_columns(self):
        raw = generate_sample_data(rows=500, seed=7)
        self.assertEqual(len(raw), 500)
        self.assertIn("Churn", raw.columns)
        self.assertIn("TotalCharges", raw.columns)

    def test_cleaning_handles_blank_total_charges(self):
        raw = generate_sample_data(rows=500, seed=9)
        cleaned = clean_telco_dataframe(raw)
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertIn("TenureBand", cleaned.columns)
        self.assertIn("ServiceCount", cleaned.columns)
        self.assertTrue(set(cleaned["Churn"].unique()).issubset({"Yes", "No"}))

    def test_customer_record_has_engineered_fields(self):
        values = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 8, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 65.0,
        }
        record = make_customer_record(values)
        self.assertIn("TenureBand", record.columns)
        self.assertIn("ServiceCount", record.columns)
        self.assertAlmostEqual(float(record.loc[0, "TotalCharges"]), 520.0)

    def test_model_features_separate_target(self):
        cleaned = clean_telco_dataframe(generate_sample_data(rows=500, seed=11))
        X, y = model_features(cleaned)
        self.assertEqual(len(X), len(y))
        self.assertNotIn("Churn", X.columns)
        self.assertEqual(set(y.unique()), {0, 1})

    def test_three_model_training_returns_scorecard(self):
        cleaned = clean_telco_dataframe(generate_sample_data(rows=600, seed=13))
        bundle = train_model_bundle(cleaned)
        self.assertEqual(set(bundle["metrics"]["Model"]), {"Logistic Regression", "Decision Tree", "Random Forest"})
        self.assertIn(bundle["best_model"], bundle["results"])
        self.assertGreaterEqual(float(bundle["results"][bundle["best_model"]]["metrics"]["ROC-AUC"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
