from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="MLOps API - Pago a Tiempo",
    version="1.0.0"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_pipeline.joblib")

TARGET = "pago_atiempo"
PIPELINE = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros en JSON para predicción batch."
    )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
    )
    return df


def fe_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = _standardize_columns(df)

    df.replace(["NA", "N/A", "null", "None", "-", ""], np.nan, inplace=True)

    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
        df["anio_prestamo"] = df["fecha_prestamo"].dt.year
        df["mes_prestamo"] = df["fecha_prestamo"].dt.month
        df.drop(columns=["fecha_prestamo"], inplace=True)

    for col in ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "puntaje" in df.columns:
        df.drop(columns=["puntaje"], inplace=True)

    if TARGET in df.columns:
        df.drop(columns=[TARGET], inplace=True)

    return df


def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def _startup_load_model():
    global PIPELINE
    PIPELINE = load_pipeline()


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "API running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": PIPELINE is not None,
        "model_file": os.path.basename(MODEL_PATH),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if PIPELINE is None:
        return {"error": "Modelo no cargado. Revisar startup."}

    df_raw = pd.DataFrame(req.records)

    if df_raw.empty:
        return {"error": "No se recibieron registros."}

    df = fe_from_raw(df_raw)

    proba = PIPELINE.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "n_records": int(len(df_raw)),
        "predictions": [
            {
                "proba_pago_atiempo": float(p),
                "pred_pago_atiempo": int(y)
            }
            for p, y in zip(proba, pred)
        ]
    }