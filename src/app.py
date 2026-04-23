from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ft_engineering import load_dataset
from model_monitoring import current_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIFT_REPORT_PATH = PROJECT_ROOT / "drift_report.csv"
DRIFT_HISTORY_PATH = PROJECT_ROOT / "drift_history.csv"
BASELINE_PATH = PROJECT_ROOT / "baseline_reference.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "feature_importance.csv"

st.set_page_config(
    page_title="MLOps Drift Monitoring",
    page_icon="📈",
    layout="wide"
)

# =====================================
# Global style
# =====================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: rgba(255,255,255,0.02);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# Helpers
# =====================================
@st.cache_data
def load_drift_report() -> pd.DataFrame:
    return pd.read_csv(DRIFT_REPORT_PATH)


@st.cache_data
def load_drift_history() -> pd.DataFrame:
    return pd.read_csv(DRIFT_HISTORY_PATH)


@st.cache_data
def load_baseline() -> pd.DataFrame:
    return pd.read_csv(BASELINE_PATH)


@st.cache_data
def load_current_batch_data() -> pd.DataFrame:
    df = load_dataset()
    df_raw = df.drop(columns=["pago_atiempo"], errors="ignore")
    return current_batch(df_raw, n=2000)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def compute_risk_score(report: pd.DataFrame) -> float:
    if report.empty:
        return 0.0
    weighted = (
        (report["overall_status"] == "DRIFT").sum() * 1.0
        + (report["overall_status"] == "WARNING").sum() * 0.5
    )
    return round((weighted / len(report)) * 100, 2)


def risk_label(score: float) -> str:
    if score < 20:
        return "LOW"
    if score < 40:
        return "MEDIUM"
    return "HIGH"


def add_priority_columns(report: pd.DataFrame) -> pd.DataFrame:
    df = report.copy()

    severity_map = {"OK": 0, "WARNING": 1, "DRIFT": 2, "NA": -1}
    df["severity_rank"] = df["overall_status"].map(severity_map).fillna(-1)

    metric_cols = [c for c in ["psi", "js"] if c in df.columns]
    if metric_cols:
        df["risk_score"] = df[metric_cols].fillna(0).sum(axis=1)
    else:
        df["risk_score"] = 0.0

    df = df.sort_values(
        by=["severity_rank", "risk_score", "variable"],
        ascending=[False, False, True]
    )
    return df


def style_status_table(df: pd.DataFrame):
    def color_status(val):
        if val == "OK":
            return "background-color: rgba(0, 200, 0, 0.18); color: #b7ffb7;"
        if val == "WARNING":
            return "background-color: rgba(255, 165, 0, 0.22); color: #ffd27f;"
        if val == "DRIFT":
            return "background-color: rgba(255, 0, 0, 0.18); color: #ff9b9b;"
        return ""

    cols = [
        c for c in ["psi_status", "js_status", "ks_status", "chi_status", "overall_status"]
        if c in df.columns
    ]
    return df.style.map(color_status, subset=cols)


def plot_numeric_distribution(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    variable: str
) -> None:
    base_series = baseline[variable].dropna()
    curr_series = current[variable].dropna()

    if base_series.empty and curr_series.empty:
        st.info("No hay datos suficientes para graficar esta variable.")
        return

    plot_df = pd.concat([
        pd.DataFrame({"value": base_series, "dataset": "baseline"}),
        pd.DataFrame({"value": curr_series, "dataset": "current"})
    ], ignore_index=True)

    fig = px.histogram(
        plot_df,
        x="value",
        color="dataset",
        barmode="overlay",
        nbins=30,
        opacity=0.65,
        title=f"Distribución: {variable}",
        color_discrete_map={"baseline": "#6baed6", "current": "#f4a259"}
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=360,
        legend_title_text=""
    )
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Frecuencia")

    st.plotly_chart(fig, use_container_width=True)


def plot_top_drift_features(report: pd.DataFrame, top_n: int = 10) -> None:
    if report.empty:
        st.info("No hay datos para mostrar.")
        return

    df = report.copy()

    if "risk_score" not in df.columns:
        metric_cols = [c for c in ["psi", "js"] if c in df.columns]
        if metric_cols:
            df["risk_score"] = df[metric_cols].fillna(0).sum(axis=1)
        else:
            df["risk_score"] = 0.0

    top_df = (
        df.sort_values("risk_score", ascending=False)
        .head(top_n)
        .sort_values("risk_score", ascending=True)
    )

    fig = px.bar(
        top_df,
        x="risk_score",
        y="variable",
        orientation="h",
        title="Top variables con mayor riesgo de drift",
        color_discrete_sequence=["#4ea8de"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
        showlegend=False
    )
    fig.update_xaxes(title_text="Risk score")
    fig.update_yaxes(title_text="")

    st.plotly_chart(fig, use_container_width=True)


def plot_drift_history(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.info("No hay historial para visualizar.")
        return

    fig = px.line(
        hist,
        x="timestamp",
        y=["ok", "warning", "drift"],
        markers=True,
        title="Alert Trend Through Time"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=360,
        legend_title_text=""
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Cantidad")

    st.plotly_chart(fig, use_container_width=True)


def plot_global_risk_gauge(score: float) -> None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Global Drift Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, 20], "color": "green"},
                {"range": [20, 40], "color": "gold"},
                {"range": [40, 100], "color": "red"},
            ],
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=320
    )

    st.plotly_chart(fig, use_container_width=True)


def get_recommended_actions(alert_df: pd.DataFrame) -> list[str]:
    actions = []

    if alert_df.empty:
        actions.append("No immediate action required. Continue monitoring.")
        return actions

    drift_vars = alert_df["variable"].tolist()

    if "tendencia_ingresos" in drift_vars:
        actions.append(
            "Review category distribution shift in 'tendencia_ingresos' and validate upstream data mapping."
        )

    if any(v in drift_vars for v in ["puntaje_datacredito", "saldo_mora", "saldo_total"]):
        actions.append(
            "Investigate credit-risk related features and assess whether retraining is needed."
        )

    if len(drift_vars) >= 3:
        actions.append(
            "Multiple features in drift: run data quality checks and consider triggering retraining."
        )

    if not actions:
        actions.append(
            "Investigate drifted features and compare source system distributions vs baseline."
        )

    return actions


def plot_importance_vs_drift(merged_df: pd.DataFrame) -> None:
    if merged_df.empty:
        st.info("No hay datos para mostrar.")
        return

    fig = px.scatter(
        merged_df,
        x="importance",
        y="risk_score",
        color="overall_status",
        hover_name="variable",
        size="importance",
        title="Feature Importance × Drift Risk",
        color_discrete_map={
            "OK": "#6baed6",
            "WARNING": "#f4a259",
            "DRIFT": "#ff6b6b"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
        legend_title_text=""
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================
# UI Header
# =====================================
st.title("MLOps Monitoring Console")
st.caption("Production-style dashboard para monitoreo de data drift y observabilidad del pipeline.")

st.markdown(
    """
    Este dashboard compara un **baseline histórico** contra un **batch actual** para detectar drift
    en variables numéricas y categóricas.

    **Métricas monitoreadas**
    - Numéricas: KS p-value, PSI, Jensen-Shannon
    - Categóricas: Chi-cuadrado p-value
    """
)

# =====================================
# Guard clause
# =====================================
if not DRIFT_REPORT_PATH.exists():
    st.warning(
        "No existe `drift_report.csv`. Ejecutá primero:\n\n"
        "`python -m src.model_monitoring`"
    )
    st.stop()

# =====================================
# Load data
# =====================================
report = load_drift_report()
report = add_priority_columns(report)

hist = load_drift_history() if DRIFT_HISTORY_PATH.exists() else pd.DataFrame()
baseline = load_baseline() if BASELINE_PATH.exists() else pd.DataFrame()
current = load_current_batch_data()
feature_importance = load_feature_importance() if FEATURE_IMPORTANCE_PATH.exists() else pd.DataFrame()

global_risk = compute_risk_score(report)
risk_text = risk_label(global_risk)

ok_count = int((report["overall_status"] == "OK").sum())
warning_count = int((report["overall_status"] == "WARNING").sum())
drift_count = int((report["overall_status"] == "DRIFT").sum())

last_run = hist["timestamp"].iloc[-1] if not hist.empty else "N/A"
n_features = len(report)

# =====================================
# Sidebar
# =====================================
st.sidebar.header("Control Panel")

show_only_alerts = st.sidebar.checkbox("Mostrar solo WARNING/DRIFT", value=False)

selected_metric = st.sidebar.selectbox(
    "Ordenar ranking por",
    ["risk_score", "psi", "js"]
)

top_n = st.sidebar.slider(
    "Top variables a mostrar",
    min_value=5,
    max_value=20,
    value=10
)

type_filter = st.sidebar.multiselect(
    "Filtrar por tipo",
    options=["numeric", "categorical"],
    default=["numeric", "categorical"]
)

status_filter = st.sidebar.multiselect(
    "Filtrar por estado",
    options=["OK", "WARNING", "DRIFT"],
    default=["OK", "WARNING", "DRIFT"]
)

csv_report = report.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Descargar drift_report.csv",
    data=csv_report,
    file_name="drift_report.csv",
    mime="text/csv"
)

if not hist.empty:
    csv_hist = hist.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Descargar drift_history.csv",
        data=csv_hist,
        file_name="drift_history.csv",
        mime="text/csv"
    )

filtered_report = report.copy()
filtered_report = filtered_report[filtered_report["type"].isin(type_filter)]
filtered_report = filtered_report[filtered_report["overall_status"].isin(status_filter)]

if show_only_alerts:
    filtered_report = filtered_report[
        filtered_report["overall_status"].isin(["WARNING", "DRIFT"])
    ]

# =====================================
# Alert banner / primary incident
# =====================================
critical = filtered_report[filtered_report["overall_status"] == "DRIFT"].head(1)

if not critical.empty:
    row = critical.iloc[0]
    st.markdown("## Primary Incident")
    st.error(
        f"""
Feature crítica: **{row['variable']}**

Tipo: **{row['type']}**

Status: **{row['overall_status']}**
"""
    )

if drift_count > 0:
    st.error("Drift detected in one or more monitored features. Review incident panel and diagnostics.")
elif warning_count > 0:
    st.warning("Moderate drift warning detected. Keep monitoring closely.")
else:
    st.success("Model stable. No significant drift detected.")

# =====================================
# KPI cards
# =====================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("OK", ok_count)
c2.metric("WARNING", warning_count)
c3.metric("DRIFT", drift_count)
c4.metric("Last Drift Run", last_run)
c5.metric("Global Risk", f"{risk_text} ({global_risk}%)")

st.markdown("---")

# =====================================
# Tabs
# =====================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Drift Analysis", "Feature Diagnostics", "History"]
)

with tab1:
    st.subheader("Executive Overview")

    st.markdown("### Global Risk Gauge")
    plot_global_risk_gauge(global_risk)

    st.markdown("### Top Drifted Features")
    overview_df = filtered_report.copy()

    if selected_metric in overview_df.columns:
        overview_df = overview_df.sort_values(by=selected_metric, ascending=False)

    top_table_cols = [c for c in ["variable", "type", "psi", "js", "overall_status", "risk_score"] if c in overview_df.columns]
    st.dataframe(
        overview_df[top_table_cols].head(top_n),
        use_container_width=True
    )

    top_alerts = overview_df[
        overview_df["overall_status"].isin(["WARNING", "DRIFT"])
    ].copy()

    if not top_alerts.empty:
        st.markdown("### Visual ranking")
        plot_top_drift_features(top_alerts, top_n=top_n)

    st.markdown("### Incident Panel")
    incidents = report[report["overall_status"] == "DRIFT"]

    if incidents.empty:
        st.success("No active incidents.")
    else:
        for _, row in incidents.iterrows():
            st.error(
                f"Feature: **{row['variable']}** | Type: **{row['type']}** | Status: **{row['overall_status']}**"
            )

    st.markdown("### Recommended Actions")
    for action in get_recommended_actions(incidents):
        st.info(action)

    st.markdown("### Impact vs Drift Matrix")
    if not feature_importance.empty:
        merged = report.merge(feature_importance, on="variable", how="inner")
        if not merged.empty:
            plot_importance_vs_drift(merged)
        else:
            st.info("No hubo match entre drift report y feature importance.")
    else:
        st.info("No existe `feature_importance.csv` todavía. Podés generarlo desde entrenamiento.")

with tab2:
    st.subheader("Drift Analysis Table")
    st.dataframe(
        style_status_table(filtered_report),
        use_container_width=True
    )

with tab3:
    st.subheader("Feature Diagnostics")

    num_vars = filtered_report.loc[
        filtered_report["type"] == "numeric", "variable"
    ].tolist()

    if num_vars:
        selected = st.selectbox("Seleccioná una variable numérica", num_vars)

        if not baseline.empty and selected in baseline.columns and selected in current.columns:
            plot_numeric_distribution(baseline, current, selected)

            detail_row = report.loc[report["variable"] == selected]
            if not detail_row.empty:
                st.markdown("### Feature summary")
                detail_cols = [c for c in [
                    "variable", "type", "psi", "psi_status", "js", "js_status",
                    "ks_pvalue", "ks_status", "chi_pvalue", "chi_status",
                    "overall_status", "risk_score"
                ] if c in detail_row.columns]
                st.dataframe(detail_row[detail_cols], use_container_width=True)
        else:
            st.info("No hay baseline disponible o la variable no existe en ambos datasets.")
    else:
        st.info("No hay variables numéricas detectadas en el reporte.")

with tab4:
    st.subheader("Evolución histórica del drift")

    if not hist.empty:
        st.dataframe(hist.tail(20), use_container_width=True)
        plot_drift_history(hist)
    else:
        st.info("Aún no existe `drift_history.csv`. Ejecutá monitoreo al menos una vez.")