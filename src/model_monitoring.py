# src/model_monitoring.py
from __future__ import annotations

import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon

from ft_engineering import load_dataset
from model_deploy import fe_from_raw  # Reutilizamos el mismo FE del deploy

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_reference.csv")
DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "drift_report.csv")
DRIFT_HISTORY_PATH = os.path.join(PROJECT_ROOT, "drift_history.csv")

TARGET = "pago_atiempo"

THRESHOLDS = {
    "psi": {"ok": 0.1, "warn": 0.25},
    "js": {"ok": 0.1, "warn": 0.2},
    "ks_p": {"ok": 0.05, "warn": 0.01},
    "chi_p": {"ok": 0.05, "warn": 0.01},
}


def _status(metric: str, value: float) -> str:
    t = THRESHOLDS[metric]
    if pd.isna(value):
        return "NA"

    if metric in ("psi", "js"):
        if value < t["ok"]:
            return "OK"
        if value < t["warn"]:
            return "WARNING"
        return "DRIFT"
    else:
        # p-values
        if value > t["ok"]:
            return "OK"
        if value > t["warn"]:
            return "WARNING"
        return "DRIFT"


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = _safe_num(ref)
    cur = _safe_num(cur)
    if len(ref) < 50 or len(cur) < 50:
        return np.nan

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan

    ref_c, _ = np.histogram(ref, bins=edges)
    cur_c, _ = np.histogram(cur, bins=edges)

    ref_p = ref_c / max(ref_c.sum(), 1)
    cur_p = cur_c / max(cur_c.sum(), 1)

    eps = 1e-6
    ref_p = np.clip(ref_p, eps, 1)
    cur_p = np.clip(cur_p, eps, 1)

    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def js_div(ref: pd.Series, cur: pd.Series, bins: int = 20) -> float:
    ref = _safe_num(ref)
    cur = _safe_num(cur)
    if len(ref) < 50 or len(cur) < 50:
        return np.nan

    comb = pd.concat([ref, cur])
    if comb.min() == comb.max():
        return np.nan

    edges = np.linspace(comb.min(), comb.max(), bins + 1)
    ref_c, _ = np.histogram(ref, bins=edges)
    cur_c, _ = np.histogram(cur, bins=edges)

    ref_p = ref_c / max(ref_c.sum(), 1)
    cur_p = cur_c / max(cur_c.sum(), 1)

    eps = 1e-6
    ref_p = np.clip(ref_p, eps, 1)
    cur_p = np.clip(cur_p, eps, 1)

    return float(jensenshannon(ref_p, cur_p))


def ks_pvalue(ref: pd.Series, cur: pd.Series) -> float:
    ref = _safe_num(ref)
    cur = _safe_num(cur)
    if len(ref) < 50 or len(cur) < 50:
        return np.nan
    return float(ks_2samp(ref, cur).pvalue)


def chi_pvalue(ref: pd.Series, cur: pd.Series, top_k: int = 50) -> float:
    ref = ref.astype(str).fillna("nan")
    cur = cur.astype(str).fillna("nan")

    ref_vc = ref.value_counts()
    cur_vc = cur.value_counts()

    total = (ref_vc.add(cur_vc, fill_value=0)).sort_values(ascending=False)
    cats = total.head(top_k).index.tolist()

    ref_c = ref.where(ref.isin(cats), other="OTHER").value_counts().sort_index()
    cur_c = cur.where(cur.isin(cats), other="OTHER").value_counts().sort_index()

    all_idx = sorted(set(ref_c.index).union(set(cur_c.index)))
    ref_c = ref_c.reindex(all_idx, fill_value=0)
    cur_c = cur_c.reindex(all_idx, fill_value=0)

    if ref_c.sum() == 0 or cur_c.sum() == 0:
        return np.nan

    expected = (ref_c / ref_c.sum()) * cur_c.sum()
    expected = expected.replace(0, 1e-6)

    return float(chisquare(f_obs=cur_c.values, f_exp=expected.values).pvalue)


def build_or_load_baseline(df_raw: pd.DataFrame, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    if os.path.exists(BASELINE_PATH):
        return pd.read_csv(BASELINE_PATH)

    base_raw = df_raw.sample(n=min(n, len(df_raw)), random_state=seed)
    base = fe_from_raw(base_raw)  # FE consistente con deploy
    base.to_csv(BASELINE_PATH, index=False)
    return base


def current_batch(df_raw: pd.DataFrame, n: int = 2000, seed: int = 7) -> pd.DataFrame:
    cur_raw = df_raw.sample(n=min(n, len(df_raw)), random_state=seed)
    cur = fe_from_raw(cur_raw)
    return cur


def detect_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    common = [c for c in baseline.columns if c in current.columns]
    baseline = baseline[common]
    current = current[common]

    num_cols = baseline.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in common if c not in num_cols]

    rows = []

    for col in num_cols:
        v_psi = psi(baseline[col], current[col])
        v_js = js_div(baseline[col], current[col])
        v_ks = ks_pvalue(baseline[col], current[col])

        rows.append({
            "variable": col,
            "type": "numeric",
            "psi": v_psi, "psi_status": _status("psi", v_psi),
            "js": v_js, "js_status": _status("js", v_js),
            "ks_pvalue": v_ks, "ks_status": _status("ks_p", v_ks),
            "chi_pvalue": np.nan, "chi_status": "NA",
        })

    for col in cat_cols:
        v_chi = chi_pvalue(baseline[col], current[col])
        rows.append({
            "variable": col,
            "type": "categorical",
            "psi": np.nan, "psi_status": "NA",
            "js": np.nan, "js_status": "NA",
            "ks_pvalue": np.nan, "ks_status": "NA",
            "chi_pvalue": v_chi, "chi_status": _status("chi_p", v_chi),
        })

    report = pd.DataFrame(rows)

    def overall(r):
        statuses = [r["psi_status"], r["js_status"], r["ks_status"], r["chi_status"]]
        if "DRIFT" in statuses:
            return "DRIFT"
        if "WARNING" in statuses:
            return "WARNING"
        if "OK" in statuses:
            return "OK"
        return "NA"

    report["overall_status"] = report.apply(overall, axis=1)
    return report.sort_values(by=["overall_status", "variable"], ascending=[False, True])


def append_history(report: pd.DataFrame) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = report["overall_status"].value_counts(dropna=False).to_dict()

    row = {
        "timestamp": ts,
        "ok": int(counts.get("OK", 0)),
        "warning": int(counts.get("WARNING", 0)),
        "drift": int(counts.get("DRIFT", 0)),
        "na": int(counts.get("NA", 0)),
    }

    hist = pd.DataFrame([row])
    if os.path.exists(DRIFT_HISTORY_PATH):
        prev = pd.read_csv(DRIFT_HISTORY_PATH)
        hist = pd.concat([prev, hist], ignore_index=True)

    hist.to_csv(DRIFT_HISTORY_PATH, index=False)


def main():
    df = load_dataset()  # trae también el target
    df_raw = df.drop(columns=[TARGET], errors="ignore")  # para drift usamos features

    baseline = build_or_load_baseline(df_raw, n=5000)
    current = current_batch(df_raw, n=2000)

    report = detect_drift(baseline, current)
    report.to_csv(DRIFT_REPORT_PATH, index=False)

    append_history(report)

    print("✅ Drift report generado:", DRIFT_REPORT_PATH)
    print(report.head(20))


if __name__ == "__main__":
    main()