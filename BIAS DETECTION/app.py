"""
GROUP 18 — Bias Detection in a Credit Scoring Model
Tanzania Credit Applicants — Interactive Streamlit Dashboard

Logistic Regression credit scoring model on anonymized Tanzanian
applicant data, auditing model for gender / region / occupation bias using
Fairlearn, testing a Reweighing mitigation strategy, and let a user score a
new applicant interactively.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

# PAGE CONFIG & STYLE
st.set_page_config(
    page_title="Tanzania coring — Bias Audit Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1B6FA8"      # blue
ACCENT = "#1F8A55"       # green
WARN = "#D64545"         # red
GOLD = "#E3A008"         # gold accent
NEUTRAL = "#3B4252"

CUSTOM_CSS = f"""
<style>
    .main {{ background-color: #F7F9FB; }}
    .block-container {{ padding-top: 1.4rem; }}

    .hero {{
        background: linear-gradient(90deg, {PRIMARY} 0%, #145586 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 6px 18px rgba(27,111,168,0.25);
    }}
    .hero h1 {{ margin: 0; font-size: 1.7rem; font-weight: 700; }}
    .hero p {{ margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #E7ECF1;
        border-radius: 12px;
        padding: 0.8rem 1rem 0.6rem 1rem;
        box-shadow: 0 2px 8px rgba(20,20,40,0.04);
    }}
    div[data-testid="stMetricLabel"] {{ font-weight: 600; color: {NEUTRAL}; }}

    .section-card {{
        background: white;
        border: 1px solid #E7ECF1;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(20,20,40,0.04);
    }}
    .flag-strip {{
        height: 6px; width: 100%;
        background: linear-gradient(90deg, #1EB53A 0%, #FFD500 33%, #00A3DD 66%, #000000 100%);
        border-radius: 6px; margin-bottom: 1.2rem;
    }}
    .verdict-good {{ color: {ACCENT}; font-weight: 700; }}
    .verdict-bad {{ color: {WARN}; font-weight: 700; }}
    .small-note {{ color: #667085; font-size: 0.85rem; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
CATEGORY_COLORS = {
    "Machinga/Street Vendor": "#E3A008",
    "Informal Trader": "#F2C14E",
    "SME Owner (Duka)": "#1F8A55",
    "Salaried/Formal": "#1B6FA8",
    "Mortgage Applicant": "#6C4AB6",
}
GENDER_COLORS = {"Female": "#D6449A", "Male": "#1B6FA8"}

# DATA LOADING
DEFAULT_PATH = "tz_credit_applicants_300000.csv"


@st.cache_data(show_spinner="Loading applicant data...")
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    df["default_label"] = df["default_label"].astype(int)
    return df


st.sidebar.markdown("### 📁 Dataset")
uploaded = st.sidebar.file_uploader("Upload applicant CSV (optional)", type="csv")

try:
    df_raw = load_data(uploaded) if uploaded is not None else load_data(DEFAULT_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_PATH}` next to app.py. "
        "Upload the CSV using the sidebar uploader."
    )
    st.stop()

REGIONS = sorted(df_raw["region"].unique().tolist())
OCCUPATIONS = sorted(df_raw["occupation_tier"].unique().tolist())
GENDERS = sorted(df_raw["gender"].unique().tolist())

# MODEL TRAINING (cached — trained once on full dataset)
@st.cache_resource(show_spinner="Training logistic regression credit model...")
def train_model(_df):
    data = _df.copy()

    le_gender = LabelEncoder().fit(data["gender"])
    le_region = LabelEncoder().fit(data["region"])
    le_occ = LabelEncoder().fit(data["occupation_tier"])

    data["gender_enc"] = le_gender.transform(data["gender"])
    data["region_enc"] = le_region.transform(data["region"])
    data["occupation_enc"] = le_occ.transform(data["occupation_tier"])

    feature_cols = [
        "age",
        "gender_enc",
        "region_enc",
        "occupation_enc",
        "monthly_income_tzs",
        "loan_amount_tzs",
    ]
    X = data[feature_cols].copy()
    y = data["default_label"].copy()

    scaler = StandardScaler()
    num_cols = ["age", "monthly_income_tzs", "loan_amount_tzs"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Bring back readable sensitive attributes aligned to the test split
    sens_test = data.loc[X_test.index, ["gender", "region", "occupation_tier"]].reset_index(drop=True)

    return {
        "model": model,
        "scaler": scaler,
        "le_gender": le_gender,
        "le_region": le_region,
        "le_occ": le_occ,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "sens_test": sens_test,
    }


bundle = train_model(df_raw)


@st.cache_resource(show_spinner="Testing Reweighing bias-mitigation strategy...")
def mitigate_bias(_bundle, _df, sensitive_col):
    """
    Reweighing (Kamiran & Calders): re-fit logistic regression with
    sample weights that balance each (sensitive-group, label) combination
    so the training signal no longer over-represents any group's outcome.
    """
    X_train, y_train = _bundle["X_train"], _bundle["y_train"]
    X_test = _bundle["X_test"]

    data = _df.copy()
    train_group = data.loc[X_train.index, sensitive_col].reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)

    n = len(y_train_r)
    weights = pd.Series(1.0, index=range(n))
    for g in train_group.unique():
        for label in [0, 1]:
            mask = (train_group == g) & (y_train_r == label)
            n_gl = mask.sum()
            if n_gl == 0:
                continue
            p_g = (train_group == g).mean()
            p_l = (y_train_r == label).mean()
            p_gl = n_gl / n
            weights.loc[mask.values] = (p_g * p_l) / p_gl

    fair_model = LogisticRegression(max_iter=1000)
    fair_model.fit(X_train, y_train, sample_weight=weights.values)

    y_pred_fair = fair_model.predict(X_test)
    y_prob_fair = fair_model.predict_proba(X_test)[:, 1]
    return y_pred_fair, y_prob_fair, fair_model


def fmt_tzs(x):
    return f"TZS {x:,.0f}"


# HERO HEADER

st.markdown(
    f"""
    <div class="hero">
        <h1>Bias Detection in a Credit Scoring Model</h1>
        <p>Logistic Regression credit scoring across Tanzania's economic spectrum. From Machinga
        street vendors to salaried mortgage applicants. Audited for gender, region and occupation bias.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="flag-strip"></div>', unsafe_allow_html=True)

# SIDEBAR — GLOBAL FILTERS
st.sidebar.markdown("###  Explore Filters")
st.sidebar.caption("Filters apply to the Overview & Exploratory tabs only. The model is always trained on the full dataset.")

f_regions = st.sidebar.multiselect("Region", REGIONS, default=REGIONS)
f_genders = st.sidebar.multiselect("Gender", GENDERS, default=GENDERS)
f_occ = st.sidebar.multiselect("Occupation tier", OCCUPATIONS, default=OCCUPATIONS)

df = df_raw[
    df_raw["region"].isin(f_regions)
    & df_raw["gender"].isin(f_genders)
    & df_raw["occupation_tier"].isin(f_occ)
]

st.sidebar.markdown("---")
st.sidebar.markdown("### Fairness lens")
sensitive_choice = st.sidebar.radio(
    "Audit the model against:",
    ["gender", "region", "occupation_tier"],
    format_func=lambda x: {"gender": "Gender", "region": "Region", "occupation_tier": "Occupation tier"}[x],
)

# TABS
tab_overview, tab_eda, tab_model, tab_fair, tab_mitigate, tab_predict = st.tabs(
    [
        "Overview",
        "Exploratory Analysis",
        "Model Performance",
        "Fairness Audit",
        "Bias Mitigation",
        "Score a New Applicant",
    ]
)

# TAB 1 — OVERVIEW
with tab_overview:
    if df.empty:
        st.warning("No applicants match the current filters. Adjust the sidebar filters.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Applicants", f"{len(df):,}")
        c2.metric("Default rate", f"{df['default_label'].mean()*100:.1f}%")
        c3.metric("Avg. monthly income", fmt_tzs(df["monthly_income_tzs"].mean()))
        c4.metric("Avg. loan amount", fmt_tzs(df["loan_amount_tzs"].mean()))
        c5.metric("Female share", f"{(df['gender']=='Female').mean()*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Applicants by occupation tier")
            occ_counts = df["occupation_tier"].value_counts().reset_index()
            occ_counts.columns = ["occupation_tier", "count"]
            fig = px.bar(
                occ_counts, x="count", y="occupation_tier", orientation="h",
                color="occupation_tier", color_discrete_map=CATEGORY_COLORS,
                template=PLOTLY_TEMPLATE, text="count",
            )
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Applicants", height=380)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("#### Applicants by region")
            reg_counts = df["region"].value_counts().reset_index()
            reg_counts.columns = ["region", "count"]
            fig = px.bar(
                reg_counts.sort_values("count"), x="count", y="region", orientation="h",
                color="count", color_continuous_scale="Blues", template=PLOTLY_TEMPLATE, text="count",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title="", xaxis_title="Applicants", height=380)
            st.plotly_chart(fig, width='stretch'

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Gender split")
            gcounts = df["gender"].value_counts().reset_index()
            gcounts.columns = ["gender", "count"]
            fig = px.pie(
                gcounts, names="gender", values="count", hole=0.55,
                color="gender", color_discrete_map=GENDER_COLORS, template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig,width='stretch')

        with col4:
            st.markdown("#### Default label balance")
            dcounts = df["default_label"].map({0: "No Default", 1: "Default"}).value_counts().reset_index()
            dcounts.columns = ["label", "count"]
            fig = px.pie(
                dcounts, names="label", values="count", hole=0.55,
                color="label", color_discrete_map={"No Default": ACCENT, "Default": WARN},
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, width='stretch')

# TAB 2 — EXPLORATORY ANALYSIS
with tab_eda:
    if df.empty:
        st.warning("No applicants match the current filters.")
    else:
        st.markdown("#### Distributions")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.histogram(df, x="age", nbins=30, color_discrete_sequence=[PRIMARY], template=PLOTLY_TEMPLATE, title="Age")
            fig.update_layout(height=320, bargap=0.05)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = px.histogram(df, x="monthly_income_tzs", nbins=30, color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE, title="Monthly income (TZS)")
            fig.update_layout(height=320, bargap=0.05)
            st.plotly_chart(fig, width='stretch')
        with c3:
            fig = px.histogram(df, x="loan_amount_tzs", nbins=30, color_discrete_sequence=[GOLD], template=PLOTLY_TEMPLATE, title="Loan amount (TZS)")
            fig.update_layout(height=320, bargap=0.05)
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### Default patterns")
        c4, c5 = st.columns(2)
        with c4:
            grp = df.groupby(["gender", "default_label"]).size().reset_index(name="count")
            grp["default_label"] = grp["default_label"].map({0: "No Default", 1: "Default"})
            fig = px.bar(
                grp, x="gender", y="count", color="default_label", barmode="group",
                color_discrete_map={"No Default": ACCENT, "Default": WARN}, template=PLOTLY_TEMPLATE,
                title="Default by gender",
            )
            fig.update_layout(height=380, legend_title="")
            st.plotly_chart(fig, width='stretch')

        with c5:
            grp = df.groupby(["occupation_tier", "default_label"]).size().reset_index(name="count")
            grp["default_label"] = grp["default_label"].map({0: "No Default", 1: "Default"})
            fig = px.bar(
                grp, x="occupation_tier", y="count", color="default_label", barmode="group",
                color_discrete_map={"No Default": ACCENT, "Default": WARN}, template=PLOTLY_TEMPLATE,
                title="Default by occupation tier",
            )
            fig.update_layout(height=380, legend_title="", xaxis_tickangle=-20)
            st.plotly_chart(fig, width='stretch')

        c6, c7 = st.columns(2)
        with c6:
            st.markdown("#### Default rate by region")
            rate = df.groupby("region")["default_label"].mean().sort_values().reset_index()
            fig = px.bar(
                rate, x="default_label", y="region", orientation="h",
                color="default_label", color_continuous_scale="Reds", template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Default rate", yaxis_title="", height=380, xaxis_tickformat=".0%")
            st.plotly_chart(fig, width='stretch')

        with c7:
            st.markdown("#### Income vs. loan amount")
            plot_df = df.sample(min(8000, len(df)), random_state=42)
            fig = px.scatter(
                plot_df, x="monthly_income_tzs", y="loan_amount_tzs",
                color=plot_df["default_label"].map({0: "No Default", 1: "Default"}),
                color_discrete_map={"No Default": ACCENT, "Default": WARN},
                opacity=0.35, template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=380, legend_title="", xaxis_title="Monthly income (TZS)", yaxis_title="Loan amount (TZS)")
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### Correlation heatmap")
        enc = df.copy()
        enc["gender"] = LabelEncoder().fit_transform(enc["gender"])
        enc["region"] = LabelEncoder().fit_transform(enc["region"])
        enc["occupation_tier"] = LabelEncoder().fit_transform(enc["occupation_tier"])
        corr_cols = ["age", "gender", "region", "occupation_tier", "monthly_income_tzs", "loan_amount_tzs", "default_label"]
        corr = enc[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template=PLOTLY_TEMPLATE, aspect="auto",
        )
        fig.update_layout(height=440)
        st.plotly_chart(fig, width='stretch')

# TAB 3 — MODEL PERFORMANCE
with tab_model:
    y_test, y_pred, y_prob = bundle["y_test"], bundle["y_pred"], bundle["y_prob"]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["No Default", "Default"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc*100:.1f}%")
    c2.metric("ROC AUC", f"{auc_score:.3f}")
    c3.metric("Precision (Default)", f"{report['Default']['precision']*100:.1f}%")
    c4.metric("Recall (Default)", f"{report['Default']['recall']*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Pred: No Default", "Pred: Default"], y=["True: No Default", "True: Default"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### ROC curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC={auc_score:.3f})", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, width='stretch')

    with st.expander("Full classification report"):
        st.dataframe(pd.DataFrame(report).transpose().round(3), width='stretch')

    with st.expander(" Model coefficients (standardized features)"):
        coef_df = pd.DataFrame({
            "feature": bundle["feature_cols"],
            "coefficient": bundle["model"].coef_[0],
        }).sort_values("coefficient")
        fig = px.bar(
            coef_df, x="coefficient", y="feature", orientation="h",
            color="coefficient", color_continuous_scale="RdBu_r", template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

# TAB 4 — FAIRNESS AUDIT
with tab_fair:
    sens_test = bundle["sens_test"][sensitive_choice]
    y_test, y_pred = bundle["y_test"].reset_index(drop=True), bundle["y_pred"]

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sens_test,
    )

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test)
    dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=sens_test)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test)

    label = {"gender": "Gender", "region": "Region", "occupation_tier": "Occupation tier"}[sensitive_choice]
    st.markdown(f"### Auditing the model by **{label}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Demographic Parity Difference", f"{dpd:.3f}", help="0 = perfectly equal approval rates across groups. Fairlearn convention: farther from 0 = more disparity.")
    c2.metric("Disparate Impact Ratio", f"{dpr:.3f}", help="Ratio of the lowest to highest group selection rate. Regulatory rule of thumb: below 0.80 signals adverse impact.")
    c3.metric("Equalized Odds Difference", f"{eod:.3f}", help="0 = equal true-positive & false-positive rates across groups.")

    verdict = dpr >= 0.80
    if verdict:
        st.markdown(f'<p class="verdict-good">✅ Disparate impact ratio ≥ 0.80 — passes the common "four-fifths rule" screen for {label.lower()}.</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="verdict-bad">⚠️ Disparate impact ratio below 0.80 — the model shows signs of disparate impact by {label.lower()}, and should not be deployed as-is.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Selection rate (approval-for-loan rate) by group")
        sel_df = mf.by_group.reset_index().rename(columns={sensitive_choice: "group"})
        fig = px.bar(
            sel_df, x="group", y="selection_rate", color="group", template=PLOTLY_TEMPLATE,
            text=sel_df["selection_rate"].map(lambda v: f"{v:.1%}"),
        )
        fig.update_layout(height=380, showlegend=False, yaxis_tickformat=".0%", yaxis_title="Selection rate", xaxis_title="")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### True Positive vs. False Positive Rate by group")
        long_df = mf.by_group.reset_index().rename(columns={sensitive_choice: "group"}).melt(
            id_vars="group", value_vars=["true_positive_rate", "false_positive_rate"],
            var_name="metric", value_name="rate",
        )
        long_df["metric"] = long_df["metric"].map({"true_positive_rate": "TPR", "false_positive_rate": "FPR"})
        fig = px.bar(
            long_df, x="group", y="rate", color="metric", barmode="group",
            color_discrete_map={"TPR": ACCENT, "FPR": WARN}, template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=380, yaxis_tickformat=".0%", legend_title="", xaxis_title="")
        st.plotly_chart(fig, width='stretch')

    with st.expander("Full by-group metric table"):
        st.dataframe(mf.by_group.round(4), width='stretch')

    st.markdown(
        """
        <div class="section-card small-note">
        <b>Reading these numbers:</b> The model predicts <i>default probability</i>, so a lower "selection rate"
        (predicted default) for a group generally means that group is more likely to be approved. Large gaps between
        groups — especially between informal-sector applicants (Machinga, Informal Trader) and formal applicants
        (Salaried, Mortgage) — are the core fairness risk this project is designed to surface: a biased model could
        systematically exclude informal workers and rural SMEs from digital credit, even when their true repayment
        behaviour doesn't warrant it.
        </div>
        """,
        unsafe_allow_html=True,
    )

# TAB 5 — BIAS MITIGATION
with tab_mitigate:
    st.markdown("### Mitigation strategy: **Reweighing** (Kamiran & Calders)")
    st.markdown(
        """
        <div class="section-card small-note">
        Before training, each applicant is given a weight so that no combination of
        (sensitive group × default label) is over- or under-represented in the training signal.
        Groups whose outcome is rare relative to their group size are up-weighted; groups whose outcome
        is common are down-weighted. The model is then re-trained on the <i>same</i> features and split,
        so any change in fairness metrics is attributable to the mitigation, not to different data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    y_pred_fair, y_prob_fair, fair_model = mitigate_bias(bundle, df_raw, sensitive_choice)
    y_test = bundle["y_test"].reset_index(drop=True)
    sens_test = bundle["sens_test"][sensitive_choice]

    # Before
    dpd_before = demographic_parity_difference(y_test, bundle["y_pred"], sensitive_features=sens_test)
    dpr_before = demographic_parity_ratio(y_test, bundle["y_pred"], sensitive_features=sens_test)
    eod_before = equalized_odds_difference(y_test, bundle["y_pred"], sensitive_features=sens_test)
    acc_before = accuracy_score(y_test, bundle["y_pred"])

    # After
    dpd_after = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=sens_test)
    dpr_after = demographic_parity_ratio(y_test, y_pred_fair, sensitive_features=sens_test)
    eod_after = equalized_odds_difference(y_test, y_pred_fair, sensitive_features=sens_test)
    acc_after = accuracy_score(y_test, y_pred_fair)

    comp = pd.DataFrame({
        "Metric": ["Accuracy", "Demographic Parity Difference (↓ better)", "Disparate Impact Ratio (↑ better, target ≥0.80)", "Equalized Odds Difference (↓ better)"],
        "Before mitigation": [acc_before, dpd_before, dpr_before, eod_before],
        "After reweighing": [acc_after, dpd_after, dpr_after, eod_after],
    })

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"#### Impact on {label.lower()} fairness")
        st.dataframe(comp.set_index("Metric").round(4), width='stretch')

    with c2:
        plot_comp = comp[comp["Metric"] != "Accuracy"].melt(id_vars="Metric", var_name="Stage", value_name="Value")
        fig = px.bar(
            plot_comp, x="Metric", y="Value", color="Stage", barmode="group",
            color_discrete_map={"Before mitigation": WARN, "After reweighing": ACCENT},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=380, xaxis_title="", legend_title="", xaxis_tickangle=-10)
        st.plotly_chart(fig, width='stretch')

    delta_acc = (acc_after - acc_before) * 100
    delta_dpr = dpr_after - dpr_before
    st.markdown(
        f"""
        <div class="section-card">
        <b>Result:</b> Reweighing changed the disparate impact ratio from <b>{dpr_before:.3f}</b> to
        <b>{dpr_after:.3f}</b> ({'+' if delta_dpr>=0 else ''}{delta_dpr:.3f}), while accuracy moved from
        <b>{acc_before*100:.1f}%</b> to <b>{acc_after*100:.1f}%</b> ({'+' if delta_acc>=0 else ''}{delta_acc:.1f} pts).
        This is the classic fairness–accuracy trade-off: mitigation strategies rarely improve both at once, so the
        acceptable trade-off should be a deliberate policy decision, not just a modelling one.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Policy discussion: excluding informal-sector workers from digital credit")
    st.markdown(
        """
        <div class="section-card small-note">
        Machinga street vendors and informal traders typically have thinner, more volatile income records than
        salaried applicants, which a purely income/loan-based model can misread as higher risk. If left unaudited,
        this model could systematically deny or under-serve exactly the population digital credit is meant to
        reach — undermining financial inclusion goals. Practical responses include: (1) monitoring disparate
        impact ratio by occupation tier as a deployment gate, not a one-off check; (2) enriching features with
        alternative data (mobile-money transaction history, group-savings records) that better reflect informal
        income; (3) applying group-aware thresholds or reweighing as demonstrated here; and (4) pairing any
        automated score with a human-reviewed appeals path for borderline informal-sector applicants.
        </div>
        """,
        unsafe_allow_html=True,
    )

# TAB 6 — SCORE A NEW APPLICANT
with tab_predict:
    st.markdown("### Enter applicant details to get a live credit risk score")
    st.caption("Uses the original (non-mitigated) model trained on the full dataset.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_age = st.slider("Age", 18, 65, 32)
            in_gender = st.selectbox("Gender", GENDERS)
        with c2:
            in_region = st.selectbox("Region", REGIONS)
            in_occupation = st.selectbox("Occupation tier", OCCUPATIONS)
        with c3:
            in_income = st.number_input("Monthly income (TZS)", min_value=50_000, max_value=10_000_000, value=500_000, step=10_000)
            in_loan = st.number_input("Requested loan amount (TZS)", min_value=50_000, max_value=100_000_000, value=2_000_000, step=50_000)

        submitted = st.form_submit_button("🔮 Score this applicant", width='stretch')

    if submitted:
        model = bundle["model"]
        scaler = bundle["scaler"]
        le_gender, le_region, le_occ = bundle["le_gender"], bundle["le_region"], bundle["le_occ"]

        row = pd.DataFrame([{
            "age": in_age,
            "gender_enc": le_gender.transform([in_gender])[0],
            "region_enc": le_region.transform([in_region])[0],
            "occupation_enc": le_occ.transform([in_occupation])[0],
            "monthly_income_tzs": in_income,
            "loan_amount_tzs": in_loan,
        }])[bundle["feature_cols"]]

        row[bundle["num_cols"]] = scaler.transform(row[bundle["num_cols"]])
        prob_default = model.predict_proba(row)[0, 1]
        pred_label = "High Risk — likely Default" if prob_default >= 0.5 else "Low Risk — likely No Default"

        colA, colB = st.columns([1, 1.3])
        with colA:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_default * 100,
                number={"suffix": "%"},
                title={"text": "Predicted default probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PRIMARY},
                    "steps": [
                        {"range": [0, 30], "color": "#DCF3E4"},
                        {"range": [30, 60], "color": "#FDF0CC"},
                        {"range": [60, 100], "color": "#FBE0DE"},
                    ],
                    "threshold": {"line": {"color": WARN, "width": 4}, "thickness": 0.8, "value": 50},
                },
            ))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width='stretch')

            if prob_default >= 0.5:
                st.error(f"**{pred_label}** ({prob_default:.1%} predicted default probability)")
            else:
                st.success(f"**{pred_label}** ({prob_default:.1%} predicted default probability)")

        with colB:
            st.markdown("#### How this applicant compares to their peer groups")
            peer_groups = {
                "Same gender": df_raw[df_raw["gender"] == in_gender]["default_label"].mean(),
                "Same region": df_raw[df_raw["region"] == in_region]["default_label"].mean(),
                "Same occupation tier": df_raw[df_raw["occupation_tier"] == in_occupation]["default_label"].mean(),
                "Overall population": df_raw["default_label"].mean(),
            }
            peer_df = pd.DataFrame({"Group": list(peer_groups.keys()), "Historical default rate": list(peer_groups.values())})
            fig = px.bar(
                peer_df, x="Historical default rate", y="Group", orientation="h",
                color="Historical default rate", color_continuous_scale="Reds", template=PLOTLY_TEMPLATE,
            )
            fig.add_vline(x=prob_default, line_dash="dash", line_color=PRIMARY, annotation_text="This applicant's score", annotation_position="top")
            fig.update_layout(height=320, coloraxis_showscale=False, xaxis_tickformat=".0%")
            st.plotly_chart(fig, width='stretch')

            st.markdown(
                """
                <div class="section-card small-note">
                This comparison is for transparency, not for decisioning — it shows whether the model's score for
                this applicant is in line with, above, or below the historical default rate of people who share
                their gender, region, or occupation tier, helping flag cases worth a human second look.
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("Built for the NA-03 project: Bias Detection in a Credit Scoring Model — anonymized sampled Tanzanian applicant data · Logistic Regression · Fairlearn")
