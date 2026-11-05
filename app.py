from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_processing import clean_telco_dataframe, make_customer_record, segment_churn_rates
from src.modeling import train_model_bundle
from src.recommendations import retention_recommendations, segment_playbook


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "telco_customer_churn_demo.csv"

st.set_page_config(
    page_title="ChurnScope | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#edf3ff; --muted:#8e9bb5; --panel:#111a2b; --line:rgba(157,179,218,.16); --cyan:#65e4dc; --orange:#ff9b70; }
html, body, [class*="css"] { font-family:'Manrope', sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, rgba(28,82,104,.22), transparent 32%), linear-gradient(145deg,#080d18 0%,#0b1220 55%,#0b1421 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(8,13,24,.72); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1322 0%,#0d1728 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { padding-top:1.55rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:#f4f7ff !important; letter-spacing:-.035em; }
p,li,label { color:#bdc9dc; }
small,.muted { color:var(--muted) !important; }
.kicker { color:var(--cyan); font:500 11px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; margin-bottom:7px; }
.hero-title { font-size:clamp(2rem,4vw,3.35rem); line-height:1.03; font-weight:800; letter-spacing:-.06em; margin:0 0 10px 0; color:#f4f7ff; }
.hero-copy { color:#9eacc4; max-width:740px; font-size:15px; line-height:1.7; }
.brand-mark { display:flex; align-items:center; gap:11px; padding:9px 4px 22px 4px; }
.brand-symbol { display:grid; place-items:center; width:38px; height:38px; border-radius:12px; background:linear-gradient(135deg,#65e4dc,#79a8ff); color:#071321 !important; font-size:19px; font-weight:900; box-shadow:0 7px 24px rgba(101,228,220,.2); }
.brand-name { color:#f5f8ff; font-weight:800; letter-spacing:-.04em; font-size:16px; }
.brand-sub { color:#8292ae; font:10px 'DM Mono',monospace; letter-spacing:.15em; text-transform:uppercase; }
.panel { background:linear-gradient(145deg,rgba(20,31,50,.94),rgba(14,24,40,.92)); border:1px solid var(--line); border-radius:18px; padding:19px 20px; box-shadow:0 14px 40px rgba(0,0,0,.12); }
.panel-label { color:#8c9ab4; font:500 10px 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; }
.panel-value { color:#f5f8ff; font-size:29px; font-weight:800; letter-spacing:-.055em; margin-top:9px; }
.panel-foot { color:#8191ac; font-size:11px; margin-top:4px; }
.section-head { margin:23px 0 13px; display:flex; align-items:end; justify-content:space-between; gap:12px; }
.section-head h3 { margin:0; font-size:19px; }
.section-note { color:#8796b0; font-size:12px; }
.tag { display:inline-block; border:1px solid rgba(101,228,220,.25); color:#8cf0e8; background:rgba(101,228,220,.07); padding:5px 9px; border-radius:99px; font:10px 'DM Mono',monospace; letter-spacing:.08em; }
.insight { border-left:3px solid var(--orange); background:rgba(255,155,112,.07); border-radius:0 12px 12px 0; padding:13px 15px; color:#d7dfed; margin:8px 0; }
.stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid rgba(101,228,220,.28); background:linear-gradient(135deg,#65e4dc,#8bbae9); color:#071321; font-weight:800; min-height:42px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#b2fff7; color:#071321; filter:brightness(1.06); }
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"],.stSlider [data-baseweb="slider"] { border-radius:9px; }
[data-testid="stMetric"] { background:rgba(18,29,48,.78); border:1px solid var(--line); border-radius:15px; padding:15px 16px; }
[data-testid="stMetricLabel"] { color:#93a2bb !important; font:500 10px 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
[data-testid="stMetricValue"] { color:#f5f8ff !important; font-weight:800; letter-spacing:-.04em; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
hr { border-color:var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:rgba(19,30,49,.7); border-radius:9px 9px 0 0; color:#9eacc4; }
.stTabs [aria-selected="true"] { color:#70e9df !important; }
[data-testid="stAlert"] { border-radius:12px; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><h3>{title}</h3><span class="section-note">{note}</span></div>',
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="kicker">{kicker}</div><div class="hero-title">{title}</div><div class="hero-copy">{copy}</div>', unsafe_allow_html=True)


def metric_columns(data: pd.DataFrame) -> None:
    churn_rate = float(data["ChurnFlag"].mean())
    churned = int(data["ChurnFlag"].sum())
    at_risk_revenue = float(data.loc[data["ChurnFlag"].eq(1), "MonthlyCharges"].sum())
    average_tenure = float(data["tenure"].mean())
    cols = st.columns(4)
    items = [
        ("CUSTOMER BASE", f"{len(data):,}", "records in the active sample"),
        ("CHURN RATE", f"{churn_rate:.1%}", f"{churned:,} customers labelled churned"),
        ("MONTHLY REVENUE AT RISK", f"${at_risk_revenue:,.0f}", "monthly charges of churned customers"),
        ("AVERAGE TENURE", f"{average_tenure:.1f} mo", "customer relationship length"),
    ]
    for col, (label, value, foot) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="panel"><div class="panel-label">{label}</div><div class="panel-value">{value}</div><div class="panel-foot">{foot}</div></div>',
                unsafe_allow_html=True,
            )


def make_eda_charts(data: pd.DataFrame) -> None:
    section_header("Where churn concentrates", "Observed patterns · not causal effects")
    col1, col2 = st.columns(2)
    with col1:
        rates = segment_churn_rates(data, "Contract")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=rates, x="Contract", y="Churn rate", color="#65e4dc", ax=ax)
        ax.set_title("Churn rate by contract", color="#eef4ff", pad=14, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(0.5, float(rates["Churn rate"].max()) * 1.18))
        ax.tick_params(colors="#9eacc4", labelrotation=8)
        ax.set_facecolor("#111a2b")
        fig.patch.set_facecolor("#111a2b")
        for spine in ax.spines.values(): spine.set_color("#26364f")
        ax.yaxis.label.set_color("#9eacc4")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        rates = segment_churn_rates(data, "TenureBand")
        order = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
        rates["TenureBand"] = pd.Categorical(rates["TenureBand"], categories=order, ordered=True)
        rates = rates.sort_values("TenureBand")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=rates, x="TenureBand", y="Churn rate", color="#ff9b70", ax=ax)
        ax.set_title("Churn rate by customer tenure", color="#eef4ff", pad=14, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(0.5, float(rates["Churn rate"].max()) * 1.18))
        ax.tick_params(colors="#9eacc4", labelrotation=8)
        ax.set_facecolor("#111a2b")
        fig.patch.set_facecolor("#111a2b")
        for spine in ax.spines.values(): spine.set_color("#26364f")
        ax.yaxis.label.set_color("#9eacc4")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        rates = segment_churn_rates(data, "InternetService")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=rates, x="InternetService", y="Churn rate", color="#91a8ff", ax=ax)
        ax.set_title("Churn rate by internet service", color="#eef4ff", pad=14, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(0.5, float(rates["Churn rate"].max()) * 1.18))
        ax.tick_params(colors="#9eacc4", labelrotation=8)
        ax.set_facecolor("#111a2b")
        fig.patch.set_facecolor("#111a2b")
        for spine in ax.spines.values(): spine.set_color("#26364f")
        ax.yaxis.label.set_color("#9eacc4")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.histplot(
            data=data, x="MonthlyCharges", hue="Churn", bins=25,
            stat="density", common_norm=False, element="step", fill=False,
            palette={"No": "#65e4dc", "Yes": "#ff9b70"}, ax=ax,
        )
        ax.set_title("Monthly charges by churn label", color="#eef4ff", pad=14, weight="bold")
        ax.set_xlabel("Monthly charges ($)")
        ax.set_ylabel("Density")
        ax.tick_params(colors="#9eacc4")
        ax.set_facecolor("#111a2b")
        fig.patch.set_facecolor("#111a2b")
        for spine in ax.spines.values(): spine.set_color("#26364f")
        ax.xaxis.label.set_color("#9eacc4")
        ax.yaxis.label.set_color("#9eacc4")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(csv_content: str) -> dict:
    raw_data = pd.read_csv(StringIO(csv_content))
    prepared = clean_telco_dataframe(raw_data)
    return train_model_bundle(prepared)


def get_model_bundle(raw_data: pd.DataFrame) -> dict:
    csv_content = raw_data.to_csv(index=False)
    with st.spinner("Training Logistic Regression, Decision Tree, and Random Forest…"):
        return cached_model_bundle(csv_content)


# Sidebar and data selection -------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-mark"><div class="brand-symbol">◈</div><div><div class="brand-name">CHURNSCOPE</div><div class="brand-sub">CUSTOMER INTELLIGENCE</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Customer Explorer", "Model Studio", "Risk Simulator", "Retention Playbook", "Data & Export"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<div class="panel-label">DATA SOURCE</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Use an IBM Telco CSV (optional)", type=["csv"], help="Your file must contain the standard Telco churn columns, including Churn.")
    st.caption("A bundled offline demo dataset is included, so the project runs without a download.")
    st.markdown("---")
    st.markdown('<div class="panel-label">PROJECT STATUS</div>', unsafe_allow_html=True)
    st.markdown('<span class="tag">LOCAL · OFFLINE READY</span>', unsafe_allow_html=True)
    st.caption("Models run on your computer. No customer data is sent to a server.")

try:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file)
        data_label = uploaded_file.name
        source_is_demo = False
    else:
        raw_df = pd.read_csv(DEFAULT_DATA)
        data_label = "Bundled generated demo dataset"
        source_is_demo = True
    df = clean_telco_dataframe(raw_df)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if len(df) < 100:
    st.error("The dataset contains fewer than 100 valid customer rows. Please use a larger Telco churn CSV.")
    st.stop()

# Page: Overview -------------------------------------------------------------
if page == "Overview":
    page_intro(
        "RETENTION ANALYTICS / 01",
        "See the signals before customers leave.",
        "An end-to-end churn intelligence workspace: explore customer behaviour, compare three classifiers, estimate individual risk, and turn model signals into practical retention actions.",
    )
    st.write("")
    if source_is_demo:
        st.info("You are viewing the bundled generated demo dataset. It mirrors the IBM Telco column structure for an offline run; it is not IBM's original customer records. You can upload the original CSV in the sidebar at any time.")
    metric_columns(df)
    make_eda_charts(df)
    section_header("Quick reads", "Start with the customer segments that deserve attention")
    contract_rates = segment_churn_rates(df, "Contract")
    tenure_rates = segment_churn_rates(df, "TenureBand")
    col1, col2 = st.columns(2)
    with col1:
        highest = contract_rates.iloc[0]
        st.markdown(f'<div class="insight"><b>Contract signal</b><br>{highest["Contract"]} currently has the highest observed churn rate at <b>{highest["Churn rate"]:.1%}</b> in this dataset.</div>', unsafe_allow_html=True)
    with col2:
        highest_tenure = tenure_rates.iloc[0]
        st.markdown(f'<div class="insight"><b>Tenure signal</b><br>The <b>{highest_tenure["TenureBand"]}</b> segment has an observed churn rate of <b>{highest_tenure["Churn rate"]:.1%}</b>.</div>', unsafe_allow_html=True)
    st.caption("These are descriptive patterns in the supplied sample, not proof that a customer attribute causes churn.")

# Page: Customer Explorer ----------------------------------------------------
elif page == "Customer Explorer":
    page_intro("CUSTOMER SEGMENTS / 02", "Explore the customer base.", "Filter the sample by contract, service, tenure, and churn label. Inspect segments before deciding where to focus retention activity.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        contract_choice = st.selectbox("Contract", ["All"] + sorted(df["Contract"].dropna().unique().tolist()))
    with col2:
        internet_choice = st.selectbox("Internet service", ["All"] + sorted(df["InternetService"].dropna().unique().tolist()))
    with col3:
        churn_choice = st.selectbox("Customer status", ["All", "Churned", "Retained"])
    with col4:
        max_tenure = st.slider("Maximum tenure", 0, 72, 72)

    filtered = df.copy()
    if contract_choice != "All": filtered = filtered[filtered["Contract"] == contract_choice]
    if internet_choice != "All": filtered = filtered[filtered["InternetService"] == internet_choice]
    if churn_choice == "Churned": filtered = filtered[filtered["ChurnFlag"] == 1]
    elif churn_choice == "Retained": filtered = filtered[filtered["ChurnFlag"] == 0]
    filtered = filtered[filtered["tenure"] <= max_tenure]

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("MATCHING CUSTOMERS", f"{len(filtered):,}")
    c2.metric("SEGMENT CHURN RATE", f"{filtered['ChurnFlag'].mean():.1%}" if len(filtered) else "—")
    c3.metric("AVERAGE MONTHLY CHARGE", f"${filtered['MonthlyCharges'].mean():.2f}" if len(filtered) else "—")
    section_header("Filtered customer records", f"Showing {min(250, len(filtered)):,} of {len(filtered):,}")
    display_columns = ["customerID", "Contract", "tenure", "InternetService", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    st.dataframe(filtered[display_columns].head(250), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered customers (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_segments.csv",
        mime="text/csv",
    )

# Page: Model Studio ---------------------------------------------------------
elif page == "Model Studio":
    page_intro("MACHINE LEARNING / 03", "Compare models. Understand trade-offs.", "Three classification approaches are trained on a stratified holdout split. Accuracy is shown alongside precision, recall, F1, and ROC-AUC so the comparison does not depend on accuracy alone.")
    bundle = get_model_bundle(raw_df)
    best = bundle["best_model"]
    best_metrics = bundle["results"][best]["metrics"]
    st.success(f"Top ROC-AUC in this run: {best} · {best_metrics['ROC-AUC']:.3f}")
    st.caption(f"Training records: {bundle['train_size']:,} · Holdout records: {bundle['test_size']:,} · Stratified split · Random state: {bundle['random_state']}")

    section_header("Model scorecard", "Higher is better · holdout evaluation")
    metrics_display = bundle["metrics"].copy()
    st.dataframe(
        metrics_display.style.format({column: "{:.3f}" for column in ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"]}),
        use_container_width=True,
        hide_index=True,
    )

    section_header("ROC curve comparison", "Threshold-independent ranking quality")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    model_colors = {"Logistic Regression": "#65e4dc", "Decision Tree": "#ff9b70", "Random Forest": "#91a8ff"}
    for name, result in bundle["results"].items():
        ax.plot(result["fpr"], result["tpr"], label=f"{name} · AUC {result['metrics']['ROC-AUC']:.3f}", linewidth=2.2, color=model_colors[name])
    ax.plot([0, 1], [0, 1], "--", color="#62718b", linewidth=1)
    ax.set_xlabel("False positive rate", color="#9eacc4")
    ax.set_ylabel("True positive rate", color="#9eacc4")
    ax.set_title("Receiver operating characteristic", color="#eef4ff", weight="bold", pad=14)
    ax.legend(frameon=False, labelcolor="#c4cde0")
    ax.tick_params(colors="#9eacc4")
    ax.set_facecolor("#111a2b")
    fig.patch.set_facecolor("#111a2b")
    for spine in ax.spines.values(): spine.set_color("#26364f")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    section_header("Best model diagnostics", best)
    col1, col2 = st.columns([1, 1.2])
    best_result = bundle["results"][best]
    with col1:
        fig, ax = plt.subplots(figsize=(5.2, 4.3))
        sns.heatmap(
            best_result["confusion_matrix"], annot=True, fmt="d", cmap="crest", cbar=False,
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax,
        )
        ax.set_title("Confusion matrix", color="#eef4ff", weight="bold", pad=12)
        ax.set_xlabel("Predicted label", color="#9eacc4")
        ax.set_ylabel("Actual label", color="#9eacc4")
        ax.tick_params(colors="#9eacc4")
        fig.patch.set_facecolor("#111a2b")
        ax.set_facecolor("#111a2b")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        report = pd.DataFrame(best_result["classification_report"]).T
        st.markdown("**Classification report**")
        st.dataframe(report.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]), use_container_width=True)
        st.caption("Churn recall estimates how many churned customers the model identified. A high score can come with more false alarms, so review precision and the confusion matrix together.")

    section_header("Feature importance analysis", "Permutation importance on original input fields")
    top_features = bundle["feature_importance"].head(12).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(top_features["Feature"], top_features["Importance"], color="#65e4dc")
    ax.set_title(f"Which fields matter most to {best}?", color="#eef4ff", weight="bold", pad=14)
    ax.set_xlabel("Mean ROC-AUC decrease after shuffling", color="#9eacc4")
    ax.tick_params(colors="#9eacc4")
    ax.set_facecolor("#111a2b")
    fig.patch.set_facecolor("#111a2b")
    for spine in ax.spines.values(): spine.set_color("#26364f")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Permutation importance measures how much model performance changes when one input field is shuffled. It describes model reliance, not a causal effect.")
    st.download_button(
        "Download model scorecard (CSV)",
        data=bundle["metrics"].to_csv(index=False).encode("utf-8"),
        file_name="churn_model_comparison.csv",
        mime="text/csv",
    )

# Page: Risk Simulator -------------------------------------------------------
elif page == "Risk Simulator":
    page_intro("INDIVIDUAL RISK / 04", "Test a customer profile.", "Enter a hypothetical customer profile to estimate churn probability using the best-performing model from the comparison. Use the result as a review aid, not an automatic decision.")
    bundle = get_model_bundle(raw_df)
    best = bundle["best_model"]
    st.markdown(f'<span class="tag">ACTIVE MODEL · {best.upper()}</span>', unsafe_allow_html=True)
    st.write("")
    with st.form("customer_risk_form"):
        left, middle, right = st.columns(3)
        with left:
            st.markdown("**01 · Customer profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_text = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with middle:
            st.markdown("**02 · Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["Yes", "No"] if phone == "Yes" else ["No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            if internet == "No":
                no_internet = ["No internet service"]
                online_security = st.selectbox("Online security", no_internet)
                online_backup = st.selectbox("Online backup", no_internet)
                device_protection = st.selectbox("Device protection", no_internet)
                tech_support = st.selectbox("Tech support", no_internet)
                streaming_tv = st.selectbox("Streaming TV", no_internet)
                streaming_movies = st.selectbox("Streaming movies", no_internet)
            else:
                online_security = st.selectbox("Online security", ["No", "Yes"])
                online_backup = st.selectbox("Online backup", ["No", "Yes"])
                device_protection = st.selectbox("Device protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])
        with right:
            st.markdown("**03 · Contract & billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 1.0)
            st.caption("Total charges are calculated as monthly charges × tenure for this hypothetical profile.")
        submitted = st.form_submit_button("Estimate churn risk", use_container_width=True)

    if submitted:
        customer_values = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_text == "Yes" else 0,
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
        }
        customer_frame = make_customer_record(customer_values)
        probability = float(bundle["results"][best]["pipeline"].predict_proba(customer_frame)[:, 1][0])
        risk_label = "LOW" if probability < 0.25 else "MODERATE" if probability < 0.50 else "HIGH" if probability < 0.70 else "VERY HIGH"
        st.write("")
        section_header("Prediction result", "Model estimate for this entered profile")
        result_col1, result_col2 = st.columns([1, 1.5])
        with result_col1:
            st.metric("ESTIMATED CHURN PROBABILITY", f"{probability:.1%}")
            st.markdown(f'<span class="tag">{risk_label} RISK</span>', unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0))
        with result_col2:
            st.markdown("**Suggested retention actions to consider**")
            for action in retention_recommendations(customer_values):
                st.markdown(f"- {action}")
        st.warning("This is an educational model estimate based on the selected dataset. It is not a guarantee, and it should not be used as the sole basis for decisions about an individual customer.")

# Page: Retention Playbook ---------------------------------------------------
elif page == "Retention Playbook":
    page_intro("ACTION DESIGN / 05", "Translate signals into service improvements.", "Use the strongest model signals and observed churn segments to shape fair, transparent retention experiments. Validate each action with real customer feedback and measured outcomes.")
    bundle = get_model_bundle(raw_df)
    top = bundle["feature_importance"].head(6)
    col1, col2 = st.columns([1.05, 1.25])
    with col1:
        section_header("Top model signals", "Permutation importance")
        st.dataframe(top.style.format({"Importance": "{:.4f}"}), use_container_width=True, hide_index=True)
    with col2:
        section_header("Retention actions", "Mapped to top signals")
        seen = set()
        for feature in top["Feature"].tolist():
            title, action = segment_playbook(feature)
            if title not in seen:
                st.markdown(f'<div class="insight"><b>{title}</b><br>{action}</div>', unsafe_allow_html=True)
                seen.add(title)
        if not seen:
            st.caption("Train the model to see data-backed action areas.")
    section_header("Segment opportunities", "Observed churn rate by customer group")
    segment_options = ["Contract", "TenureBand", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
    selected_segment = st.selectbox("Choose a segment", segment_options)
    rates = segment_churn_rates(df, selected_segment)
    rates["Churn rate"] = rates["Churn rate"].map(lambda value: f"{value:.1%}")
    st.dataframe(rates, use_container_width=True, hide_index=True)
    st.caption("A practical next step is to run a small, opt-in retention experiment and measure customer outcomes. Do not assume model importance proves causation.")

# Page: Data & Export --------------------------------------------------------
elif page == "Data & Export":
    page_intro("DATA QUALITY / 06", "Know what the model is learning from.", "Review the cleaned table, download the prepared data, and check the exact source file used in this session.")
    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROWS", f"{len(df):,}")
    col2.metric("RAW FIELDS", f"{len(raw_df.columns):,}")
    col3.metric("MISSING VALUES AFTER CLEANING", f"{int(df.isna().sum().sum()):,}")
    if source_is_demo:
        st.warning("Bundled data status: generated educational demo data. It follows the Telco dataset's 21-column schema, but it is not the official IBM CSV.")
    else:
        st.success(f"Active data file: {data_label}")
    section_header("Cleaned data preview", "First 30 rows")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    section_header("Column dictionary", "Field names and inferred data types")
    schema = pd.DataFrame({"Column": df.columns, "Data type": [str(dtype) for dtype in df.dtypes], "Missing": [int(value) for value in df.isna().sum().values]})
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_telco_customer_churn.csv",
        mime="text/csv",
    )
    st.markdown("**Official sample dataset reference:** [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d)")
    st.caption("The optional download script in this project attempts to retrieve IBM's original CSV and validate its columns. If internet access is unavailable, the bundled demo remains usable.")

st.markdown("---")
st.caption("CHURNSCOPE · Customer Churn Prediction & Retention Analysis · Educational analytics workspace")
