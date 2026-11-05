DATASET INFORMATION

Bundled file: telco_customer_churn_demo.csv
Rows: 7,043
Columns: 21 (same core column names and data layout as the public IBM Telco Customer Churn sample)

IMPORTANT:
This bundled file is generated educational demo data. It is NOT IBM's original customer dataset and it does not contain real customer records. It was included so the project runs offline immediately.

To replace it with IBM's public sample CSV (requires an internet connection), open PowerShell or Command Prompt in the project folder and run:
    py download_ibm_dataset.py

The script validates the expected columns and row count before replacing the bundled demo CSV. If the download fails, the existing demo file is kept.

IBM source repository:
https://github.com/IBM/telco-customer-churn-on-icp4d
