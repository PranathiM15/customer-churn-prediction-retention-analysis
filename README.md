# ChurnScope — Customer Churn Prediction & Retention Analysis

A terminal-friendly Python web application for exploring customer churn, comparing classification models, estimating churn risk for a hypothetical customer, and turning model signals into practical retention ideas.

## What the app includes

- Cleaned customer churn dataset with missing-value handling.
- Exploratory data analysis (EDA): churn by contract, tenure, internet service, and monthly charges.
- Feature engineering: tenure bands and subscribed-service count.
- Categorical encoding and numeric scaling inside a scikit-learn preprocessing pipeline.
- Three classifiers: Logistic Regression, Decision Tree, and Random Forest.
- Model scorecard: accuracy, precision, recall, F1 score, and ROC-AUC.
- Confusion matrix, classification report, and ROC curve comparison.
- Permutation-based feature importance.
- Customer churn risk simulator and tailored retention suggestions.
- Customer filtering, cleaned-data export, and model-scorecard export.

## Project structure

```text
customer-churn-prediction-retention-analysis/
│  app.py
│  requirements.txt
│  README.md
│  run_app.bat
│  run_app.ps1
│  download_ibm_dataset.py
│  generate_demo_data.py
│  .gitignore
│
├─ data/
│  ├─ telco_customer_churn_demo.csv
│  └─ README.txt
│
├─ src/
│  ├─ __init__.py
│  ├─ data_processing.py
│  ├─ modeling.py
│  ├─ recommendations.py
│  └─ sample_data.py
│
└─ tests/
   └─ test_project.py
```

## 1. Requirements

- Windows 10/11
- Python 3.10 or newer (Python 3.12 is a good general-purpose choice)
- Internet connection for installing packages. The app data itself is bundled, so it can run offline after installation.

Check Python from PowerShell:

```powershell
py --version
```

## 2. Extract the ZIP

1. Download the ZIP file.
2. Right-click it and choose **Extract All**.
3. Open the extracted `customer-churn-prediction-retention-analysis` folder.
4. In the folder address bar, type `powershell` and press Enter. This opens PowerShell in the correct project folder.

Do not move `app.py` out of the folder. Keep the `data` and `src` folders beside it.

## 3. Run the application (recommended command sequence)

Copy and run these commands one at a time in PowerShell:

```powershell
py -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A browser tab should open with **ChurnScope**. If the browser does not open automatically, copy the `Local URL` shown in PowerShell (normally `http://localhost:8501`) into Chrome.

### One-click Windows option

Double-click `run_app.bat`. It creates the virtual environment if needed, installs the requirements, and starts the app. The first package installation needs internet access.

## 4. Optional: use IBM's original CSV

The bundled dataset is a generated 7,043-row educational sample that follows the original IBM Telco data's core schema. It is included for an immediate offline start and is **not** IBM's original customer records.

To attempt to download the original public IBM sample, run this in the project folder:

```powershell
.\.venv\Scripts\python.exe download_ibm_dataset.py
```

If the download succeeds, the script replaces the demo CSV after checking the expected columns and row count. If it fails, the bundled demo remains in place. You can also upload a compatible Telco churn CSV from the app's sidebar.

Source repository: https://github.com/IBM/telco-customer-churn-on-icp4d

## 5. How to use the app

1. **Overview** — review the customer base, churn rate, revenue represented by churned accounts, and segment patterns.
2. **Customer Explorer** — filter customers by contract, internet service, status, and tenure.
3. **Model Studio** — train and compare Logistic Regression, Decision Tree, and Random Forest. Review the scorecard, ROC curves, confusion matrix, classification report, and feature importance.
4. **Risk Simulator** — enter a hypothetical customer profile and estimate churn probability.
5. **Retention Playbook** — view model signals, segment rates, and practical customer-experience actions.
6. **Data & Export** — inspect cleaned data and download the prepared CSV.

The first visit to Model Studio, Risk Simulator, or Retention Playbook trains the models. Later visits reuse the cached models until the app is restarted or the dataset changes.

## 6. Stop the app

Click inside the PowerShell window and press `Ctrl + C`.

## 7. Push the project to GitHub

Your GitHub username from the supplied profile screenshot is `PranathiM15`.

### A. Create a new repository

Open: **https://github.com/new**

Enter:

- Repository name: `customer-churn-prediction-retention-analysis`
- Description: `Python web app for customer churn prediction, model comparison, and retention analysis.`
- Select **Public** (or Private if your course instructions require it).
- Do not select **Add a README**, **Add .gitignore**, or **Choose a license** because this folder already contains project files.
- Click **Create repository**.

Your repository URL should then be:

```text
https://github.com/PranathiM15/customer-churn-prediction-retention-analysis
```

### B. Check that Git is installed

```powershell
git --version
```

If PowerShell says `git` is not recognized, install Git for Windows from https://git-scm.com/download/win, finish the installer, close PowerShell, reopen it in the project folder, and run `git --version` again.

### C. Configure Git once (use your own GitHub email)

```powershell
git config --global user.name "PranathiM15"
```

```powershell
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Replace `YOUR_GITHUB_EMAIL` with the email connected to your GitHub account. Do not type the placeholder literally.

### D. Push the project

Make sure PowerShell is in the project folder, then run:

```powershell
git init
```

```powershell
git branch -M main
```

```powershell
git add .
```

```powershell
git status
```

Check that the project files and `data/telco_customer_churn_demo.csv` are listed. The `.venv` folder should not appear because it is excluded by `.gitignore`.

```powershell
git commit -m "Add customer churn prediction and retention analysis app"
```

```powershell
git remote add origin https://github.com/PranathiM15/customer-churn-prediction-retention-analysis.git
```

```powershell
git push -u origin main
```

GitHub may open a browser sign-in window. Sign in to your own account and authorize Git Credential Manager if prompted.

### E. If you see `remote origin already exists`

Run:

```powershell
git remote set-url origin https://github.com/PranathiM15/customer-churn-prediction-retention-analysis.git
```

Then push again:

```powershell
git push -u origin main
```

### F. If the first push is rejected

If you accidentally created the GitHub repository with a README or other file, run:

```powershell
git pull origin main --allow-unrelated-histories
```

Resolve any merge message by saving and closing the editor, then run:

```powershell
git push -u origin main
```

Do not use `--force` for this beginner workflow.

## 8. Troubleshooting

**`py` is not recognized** — install Python from https://www.python.org/downloads/windows/, enable the option to add Python to PATH, reopen PowerShell, and check `py --version`.

**`No module named streamlit`** — run `\.venv\Scripts\python.exe -m pip install -r requirements.txt` from the project folder.

**Port 8501 is already in use** — run `\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` and open `http://localhost:8502`.

**`git` is not recognized** — install Git for Windows, reopen PowerShell, and run `git --version`.

**Dataset upload error** — check that the CSV is a Telco churn dataset with a `Churn` column and the standard service, contract, billing, and customer columns listed in `data_processing.py`.

**Model training is taking longer than expected** — the first model run trains three classifiers and calculates permutation importance. Wait for the spinner to finish; the trained models are cached during the session.

## Academic-use note

The included demo data and models are for learning and demonstration. Model scores are dependent on the supplied data and split. Churn predictions are estimates, not guarantees or evidence that a customer attribute causes churn. Review model errors and customer context before considering any retention action.
