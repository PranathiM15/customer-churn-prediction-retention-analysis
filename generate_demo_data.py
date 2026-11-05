from pathlib import Path

from src.sample_data import generate_sample_data

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT = DATA_DIR / "telco_customer_churn_demo.csv"

generate_sample_data().to_csv(OUTPUT, index=False)
print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
