"""
Customer Churn — Decision Support Dashboard
--------------------------------------------
Run locally (in VS Code) after copying the `artifacts/` folder produced by the
Colab notebook (churn_bi_framework.ipynb) into this same directory.

    pip install -r requirements.txt
    streamlit run app.py

This app does NOT retrain anything — it only loads the final stacked ensemble
(5 tuned base models + LightGBM meta-learner), processed data, precomputed
SHAP values, and the full model-comparison table exported from Colab, and
serves them as an interactive BI / decision-support dashboard.

Final model: Stacked Ensemble (XGBoost + LightGBM + CatBoost + Random Forest +
Logistic Regression, SMOTEENN-balanced, Optuna-tuned, 5-fold OOF stacking with
a LightGBM meta-learner, F1-optimized decision threshold subject to accuracy >= 85%).
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Decision Support Dashboard", layout="wide")

ARTIFACTS_DIR = "artifacts"


# ---------------------------------------------------------------------------
# Load artifacts (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(f"{ARTIFACTS_DIR}/dashboard_data.csv")
    with open(f"{ARTIFACTS_DIR}/shap_feature_names.json") as f:
        feature_names = json.load(f)
    shap_values = np.load(f"{ARTIFACTS_DIR}/shap_values.npy")
    shap_base_value = np.load(f"{ARTIFACTS_DIR}/shap_base_value.npy")
    with open(f"{ARTIFACTS_DIR}/final_threshold.json") as f:
        final_threshold = json.load(f)["threshold"]
    model_comparison = pd.read_csv(f"{ARTIFACTS_DIR}/model_comparison.csv")

    # The exported dashboard_data.csv only has the raw columns -- the 4 engineered
    # features used by the model (and referenced by feature_names / shap_values)
    # were computed on a separate copy in the notebook and never saved to the CSV.
    # Recompute them here so df has every column feature_names expects.
    df["engagement_score"] = (df["monthly_logins"] * df["weekly_active_days"]) / (df["tenure_months"] + 1)
    df["support_burden"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["revenue_per_login"] = df["total_revenue"] / (df["monthly_logins"] + 1)
    df["low_satisfaction_flag"] = ((df["csat_score"] <= 2) & (df["nps_score"] <= 0)).astype(int)

    return df, feature_names, shap_values, shap_base_value, final_threshold, model_comparison


df, feature_names, shap_values, shap_base_value, final_threshold, model_comparison = load_data()

# Final model's own headline metrics (pulled straight from the comparison table,
# not hardcoded, so this stays correct if the notebook is rerun with different data)
final_row = model_comparison[model_comparison["phase"] == "4. Proposed Final Model"].iloc[0]
base_paper_row = model_comparison[model_comparison["model"] == "Blended Ensemble (base paper, proposed)"].iloc[0]

# ---------------------------------------------------------------------------
# Sidebar — filters (this is the "decision support" control panel)
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

slider_options = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
default_risk_tier = min(slider_options, key=lambda opt: abs(opt - float(final_threshold)))

risk_tier = st.sidebar.select_slider(
    "Minimum churn risk (probability)",
    options=slider_options,
    value=default_risk_tier,
)

segment_options = ["All"] + sorted(df["customer_segment"].dropna().unique().tolist())
segment_filter = st.sidebar.selectbox("Customer segment", segment_options)

cluster_options = ["All"] + sorted(df["cluster"].dropna().unique().tolist())
cluster_filter = st.sidebar.selectbox("Behavioral cluster", cluster_options)

contract_options = ["All"] + sorted(df["contract_type"].dropna().unique().tolist())
contract_filter = st.sidebar.selectbox("Contract type", contract_options)

st.sidebar.divider()
st.sidebar.caption(
    f"Final model decision threshold: **{final_threshold:.3f}** "
    "(F1-optimized subject to accuracy \u2265 85%)"
)

filtered = df[df["churn_probability"] >= risk_tier].copy()
if segment_filter != "All":
    filtered = filtered[filtered["customer_segment"] == segment_filter]
if cluster_filter != "All":
    filtered = filtered[filtered["cluster"] == cluster_filter]
if contract_filter != "All":
    filtered = filtered[filtered["contract_type"] == contract_filter]

# ---------------------------------------------------------------------------
# Header KPIs
# ---------------------------------------------------------------------------
st.title("📊 Customer Churn — Decision Support Dashboard")
st.caption(
    "AI-Driven BI Framework | Enriched features + clustering + SMOTEENN + "
    "Optuna-tuned stacked ensemble + SHAP explainability"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Customers (filtered)", f"{len(filtered):,}")
col2.metric("Overall churn rate", f"{df['churn'].mean()*100:.1f}%")
col3.metric(
    "Avg churn probability (filtered)",
    f"{filtered['churn_probability'].mean()*100:.1f}%" if len(filtered) else "—",
)
col4.metric(
    "Revenue at risk",
    f"${filtered.loc[filtered['churn_prediction']==1, 'total_revenue'].sum():,.0f}",
)
col5.metric(
    "Final model accuracy / F1",
    f"{final_row['accuracy']*100:.1f}% / {final_row['f1']*100:.1f}%",
)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_customers, tab_explain, tab_segments, tab_models = st.tabs(
    ["Overview", "At-Risk Customers", "Why? (SHAP Explanation)", "Segments", "Model Performance"]
)

# --- Overview tab -----------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn rate by cluster")
        cluster_churn = df.groupby("cluster")["churn"].mean().reset_index()
        st.bar_chart(cluster_churn.set_index("cluster"))

    with c2:
        st.subheader("Churn rate by customer segment")
        seg_churn = df.groupby("customer_segment")["churn"].mean().reset_index()
        st.bar_chart(seg_churn.set_index("customer_segment"))

    st.subheader("Global drivers of churn (SHAP importance, XGBoost proxy model)")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, df[feature_names], plot_type="bar", show=False
    )
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "The deployed prediction comes from a 5-model stacked ensemble; SHAP is "
        "computed on its strongest individual tree-based component (tuned XGBoost) "
        "as an interpretable proxy for what drives the ensemble's decisions."
    )

# --- At-risk customers tab ---------------------------------------------------
with tab_customers:
    st.subheader(f"Customers at or above {risk_tier:.0%} churn risk")
    show_cols = [
        "customer_id", "customer_segment", "cluster", "tenure_months",
        "monthly_fee", "total_revenue", "support_tickets", "csat_score",
        "nps_score", "churn_probability", "churn_prediction",
    ]
    st.dataframe(
        filtered[show_cols].sort_values("churn_probability", ascending=False),
        use_container_width=True,
        height=500,
    )
    st.download_button(
        "Download filtered list (CSV)",
        filtered[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="at_risk_customers.csv",
        mime="text/csv",
    )

# --- Individual explanation tab ---------------------------------------------
with tab_explain:
    st.subheader("Explain an individual customer's churn prediction")
    customer_ids = filtered["customer_id"].tolist() or df["customer_id"].tolist()
    selected_id = st.selectbox("Select customer_id", customer_ids)

    row_idx = df.index[df["customer_id"] == selected_id][0]
    row = df.loc[row_idx]

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability (ensemble)", f"{row['churn_probability']*100:.1f}%")
    c2.metric("Predicted", "Churn" if row["churn_prediction"] == 1 else "Retained")
    c3.metric("Cluster / Segment", f"{row['cluster']} / {row['customer_segment']}")

    st.write("**Why this prediction — top contributing factors (XGBoost proxy):**")
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=shap_base_value if np.ndim(shap_base_value) == 0 else shap_base_value[row_idx],
        data=df.loc[row_idx, feature_names].values,
        feature_names=feature_names,
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig2, clear_figure=True)

    st.info(
        "Use this to decide the retention action: e.g. high `support_burden` and low "
        "`csat_score` → route to support recovery; low `engagement_score` and no "
        "recent login → send a re-engagement offer; high `revenue` + high risk → "
        "priority outreach from account manager."
    )

# --- Segments tab -------------------------------------------------------------
with tab_segments:
    st.subheader("Cluster profiles")
    profile_cols = [
        "tenure_months", "monthly_logins", "total_revenue", "support_tickets",
        "csat_score", "nps_score", "churn",
    ]
    cluster_profile = df.groupby("cluster")[profile_cols].mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)
    st.caption(
        "Compare against Table 7 in the RFM+DEC base paper (Ibrahim et al., 2026) — "
        "here segments are formed from enriched behavioral, support, and satisfaction "
        "features rather than RFM alone."
    )

# --- Model Performance tab ---------------------------------------------------
with tab_models:
    st.subheader("Full model comparison — all approaches tried")
    st.dataframe(
        model_comparison.style.format({
            "accuracy": "{:.1%}", "precision": "{:.1%}",
            "recall": "{:.1%}", "f1": "{:.1%}", "roc_auc": "{:.3f}",
        }),
        use_container_width=True,
        height=550,
    )

    st.subheader("Headline comparison: this project vs. the base paper")
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Accuracy", f"{final_row['accuracy']*100:.2f}%",
               f"{(final_row['accuracy']-base_paper_row['accuracy'])*100:+.2f} pts vs base paper")
    hc2.metric("Precision", f"{final_row['precision']*100:.1f}%",
               f"{(final_row['precision']-base_paper_row['precision'])*100:+.1f} pts vs base paper")
    hc3.metric("Recall", f"{final_row['recall']*100:.1f}%",
               f"{(final_row['recall']-base_paper_row['recall'])*100:+.1f} pts vs base paper")
    hc4.metric("F1-score", f"{final_row['f1']*100:.2f}%",
               f"{(final_row['f1']-base_paper_row['f1'])*100:+.2f} pts vs base paper")

    st.caption(
        "Base paper: Ari, D. & Burukanli, M. (2025). *Blended Ensemble Learning "
        "Framework for Customer Churn Prediction in Subscription-Based Businesses.* "
        "Evaluated on the same Kaggle dataset (10,000 customers, 8,979 non-churn / "
        "1,021 churn). This project's final model is a 5-fold out-of-fold stacked "
        "ensemble (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression "
        "base learners + LightGBM meta-learner) with a decision threshold chosen to "
        "maximize F1 subject to accuracy \u2265 85%."
    )

    st.subheader("Accuracy vs. F1 — why this threshold was chosen")
    st.markdown(
        "- Naive baseline (always predict *no churn*): **~89.8% accuracy, 0% recall** "
        "— useless for retention, included only to show why accuracy alone is misleading here.\n"
        "- Maximizing F1 with no accuracy floor reaches a higher F1 but drops accuracy "
        "into the high-70s / low-80s.\n"
        "- The deployed threshold represents the best F1 achievable while still keeping "
        "accuracy at or above 85%, balancing overall correctness with the ability to "
        "actually catch at-risk customers."
    )
