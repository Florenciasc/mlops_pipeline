from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

# ✅ APP SIEMPRE DEFINIDA A NIVEL MÓDULO (esto evita "app not found")
app = FastAPI(
    title="MLOps API - Pago a Tiempo",
    version="1.0.0"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_pipeline.joblib")

TARGET = "pago_atiempo"
PIPELINE = None  # se carga en startup


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Lista de registros en JSON (batch).")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_")
    )
    return df


def fe_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering consistente con el entrenamiento, sobre JSON crudo."""
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

    # Por si viene target
    if TARGET in df.columns:
        df.drop(columns=[TARGET], inplace=True)

    return df


def train_and_save_pipeline():
    """
    Entrena modelo final (RandomForest sin 'puntaje') y guarda el pipeline.
    Importamos sklearn y tus helpers *adentro* para no romper el import del módulo.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    # Import diferido (evita que falle antes de crear app)
    from src.ft_engineering import prepare_training_data

    split, preprocessor, _ = prepare_training_data(target=TARGET, drop_cols=["puntaje"])

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(split.X_train, split.y_train)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


def load_or_train_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_pipeline()


@app.on_event("startup")
def _startup_load_model():
    global PIPELINE
    PIPELINE = load_or_train_pipeline()


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
            {"proba_pago_atiempo": float(p), "pred_pago_atiempo": int(y)}
            for p, y in zip(proba, pred)
        ]
    }