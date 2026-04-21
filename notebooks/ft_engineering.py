from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



# DATA STRUCTURE

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series



# UTILITIES

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_")
    )
    return df


# LOAD + FEATURE ENGINEERING

def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carga dataset y aplica ingeniería básica de características.
    """

    if data_path is None:
        cwd = os.getcwd()
        if cwd.endswith("src"):
            data_path = os.path.join("..", "Base_de_datos.xlsx")
        else:
            data_path = "Base_de_datos.xlsx"

    df = pd.read_excel(data_path)

    # Normalizar nombres
    df = standardize_columns(df)

    # Unificar representaciones de nulos
    df.replace(["NA", "N/A", "null", "None", "-", ""], np.nan, inplace=True)

    # FEATURE ENGINEERING FECHA
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
        df["anio_prestamo"] = df["fecha_prestamo"].dt.year
        df["mes_prestamo"] = df["fecha_prestamo"].dt.month
        df.drop(columns=["fecha_prestamo"], inplace=True)

    # FORZAR CATEGÓRICAS A STRING
    categorical_force = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

    for col in categorical_force:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# FEATURE GROUPS

def get_feature_groups(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """Identifica columnas numéricas y categóricas."""
    features = [c for c in df.columns if c != target]

    num_cols = df[features].select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    cat_cols = df[features].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return num_cols, cat_cols

# PREPROCESSOR

def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Construye pipeline de preprocesamiento."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor

# SPLIT

def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> DataSplit:
    """Train-test split estratificado."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe.")

    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return DataSplit(X_train, X_test, y_train, y_test)

# MAIN PREPARATION FUNCTION
def prepare_training_data(
    target: str = "pago_atiempo",
    data_path: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[DataSplit, ColumnTransformer, List[str]]:
    """
    Pipeline completo:
    - Load
    - Feature engineering
    - Split
    - Preprocessor fit
    """

    df = load_dataset(data_path=data_path)

    # Eliminar columnas opcionales (ej: puntaje para testear leakage)
    if drop_cols:
        drop_cols = [c.lower().strip().replace(" ", "_") for c in drop_cols]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    num_cols, cat_cols = get_feature_groups(df, target=target)

    split = split_data(df, target=target)

    preprocessor = make_preprocessor(num_cols, cat_cols)

    # Fit solo con train
    preprocessor.fit(split.X_train, split.y_train)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return split, preprocessor, feature_names