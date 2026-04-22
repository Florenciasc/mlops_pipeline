from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

THRESHOLD = 0.50
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="MLOps API - Pago a Tiempo",
    version=MODEL_VERSION,
    description="API para predecir la probabilidad de pago a tiempo de clientes a partir de variables financieras, demográficas y crediticias."
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_pipeline.joblib")

TARGET = "pago_atiempo"
PIPELINE = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros en JSON para predicción batch.",
        example=[
            {
                "edad_cliente": 35,
                "salario_cliente": 80000,
                "tipo_credito": "consumo",
                "capital_prestado": 15000,
                "plazo_meses": 12,
                "tipo_laboral": "dependiente",
                "total_otros_prestamos": 2,
                "cuota_pactada": 1800,
                "puntaje_datacredito": 720,
                "cant_creditosvigentes": 1,
                "huella_consulta": 0,
                "saldo_mora": 0,
                "saldo_principal": 12000,
                "saldo_total": 15000,
                "saldo_mora_codeudor": 0,
                "creditos_sectorfinanciero": 1,
                "creditos_sectorcooperativo": 0,
                "creditos_sectorreal": 0,
                "promedio_ingresos_datacredito": 78000,
                "tendencia_ingresos": "estable",
                "fecha_prestamo": "2024-03-15"
            }
        ]
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
    """
    Aplica la misma lógica básica de feature engineering usada antes del pipeline:
    - normaliza nombres
    - trata nulos frecuentes
    - transforma fecha_prestamo en anio_prestamo y mes_prestamo
    - elimina columnas que no deben entrar al modelo
    """
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

    # Evitar leakage
    if "puntaje" in df.columns:
        df.drop(columns=["puntaje"], inplace=True)

    # Por si el usuario envía el target
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
        "docs": "/docs",
        "model_version": MODEL_VERSION
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": PIPELINE is not None,
        "model_file": os.path.basename(MODEL_PATH),
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if PIPELINE is None:
        return {"error": "Modelo no cargado. Revisar startup."}

    df_raw = pd.DataFrame(req.records)

    if df_raw.empty:
        return {"error": "No se recibieron registros."}

    try:
        df = fe_from_raw(df_raw)

        proba = PIPELINE.predict_proba(df)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)

        return {
            "n_records": int(len(df_raw)),
            "threshold": THRESHOLD,
            "model_version": MODEL_VERSION,
            "predictions": [
                {
                    "proba_pago_atiempo": float(p),
                    "pred_pago_atiempo": int(y)
                }
                for p, y in zip(proba, pred)
            ]
        }

    except Exception as e:
        return {
            "error": "Falló la predicción",
            "detail": str(e)
        }