# src/app.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "drift_report.csv")
DRIFT_HISTORY_PATH = os.path.join(PROJECT_ROOT, "drift_history.csv")
BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_reference.csv")

# Reutilizamos monitoring para reconstruir current batch para gráficos
sys.path.append("src")
from ft_engineering import load_dataset
from model_deploy import fe_from_raw
from model_monitoring import current_batch

st.set_page_config(page_title="Data Drift Monitoring", layout="wide")
st.title("📈 Monitoreo de Data Drift (MLOps)")

st.markdown("""
Visualización del drift entre un **baseline histórico** y un **batch actual**.
Métricas:
- Numéricas: KS p-value, PSI, Jensen–Shannon
- Categóricas: Chi-cuadrado p-value
""")

if not os.path.exists(DRIFT_REPORT_PATH):
    st.warning("No existe drift_report.csv. Ejecutá primero: `python src/model_monitoring.py`")
    st.stop()

report = pd.read_csv(DRIFT_REPORT_PATH)

c1, c2, c3 = st.columns(3)
c1.metric("✅ OK", int((report["overall_status"] == "OK").sum()))
c2.metric("⚠️ WARNING", int((report["overall_status"] == "WARNING").sum()))
c3.metric("🚨 DRIFT", int((report["overall_status"] == "DRIFT").sum()))

st.subheader("📋 Tabla de drift por variable")
st.dataframe(report, use_container_width=True)

st.subheader("📊 Comparación de distribuciones (numéricas)")
num_vars = report.loc[report["type"] == "numeric", "variable"].tolist()
if num_vars:
    selected = st.selectbox("Seleccioná variable", num_vars)

    # reconstruimos baseline y current solo para graficar
    baseline = pd.read_csv(BASELINE_PATH) if os.path.exists(BASELINE_PATH) else None

    df = load_dataset()
    df_raw = df.drop(columns=["pago_atiempo"], errors="ignore")
    current = current_batch(df_raw, n=2000)

    if baseline is not None and selected in baseline.columns and selected in current.columns:
        fig = plt.figure()
        plt.hist(baseline[selected].dropna(), bins=30, alpha=0.6, label="baseline")
        plt.hist(current[selected].dropna(), bins=30, alpha=0.6, label="current")
        plt.title(f"Distribución: {selected}")
        plt.legend()
        st.pyplot(fig)
else:
    st.info("No hay variables numéricas detectadas en el reporte.")

st.subheader("📉 Evolución del drift (histórico)")
if os.path.exists(DRIFT_HISTORY_PATH):
    hist = pd.read_csv(DRIFT_HISTORY_PATH)
    st.dataframe(hist.tail(20), use_container_width=True)

    fig2 = plt.figure()
    plt.plot(hist["timestamp"], hist["ok"], label="ok")
    plt.plot(hist["timestamp"], hist["warning"], label="warning")
    plt.plot(hist["timestamp"], hist["drift"], label="drift")
    plt.xticks(rotation=45, ha="right")
    plt.title("Tendencia de alertas en el tiempo")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Aún no existe drift_history.csv. Ejecutá monitoreo al menos una vez.")