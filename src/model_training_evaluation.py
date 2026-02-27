from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ft_engineering import prepare_training_data


def summarize_classification(y_true, y_pred, y_proba) -> dict:
    """Métricas clave para dataset desbalanceado."""
    return {
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def build_model(model, preprocessor) -> Pipeline:
    """Pipeline: preprocess + modelo."""
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])


def train_eval_one(name: str, model, preprocessor, X_train, y_train, X_test, y_test) -> dict:
    pipe = build_model(model, preprocessor)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)  # umbral base (se puede ajustar luego)

    metrics = summarize_classification(y_test, y_pred, y_proba)
    metrics["model"] = name
    return metrics


def plot_comparison(df_metrics: pd.DataFrame, title: str) -> None:
    df_plot = df_metrics.set_index("model")[["roc_auc", "pr_auc", "f1_pos", "recall_pos"]]
    ax = df_plot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def run_scenario(scenario_name: str, drop_cols=None) -> pd.DataFrame:
    TARGET = "pago_atiempo"

    split, preprocessor, feature_names = prepare_training_data(
        target=TARGET,
        drop_cols=drop_cols
    )

    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    # Modelos iniciales
    models = {
        "LogReg_balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss"
        )
    }

    rows = []
    for name, model in models.items():
        rows.append(train_eval_one(name, model, preprocessor, X_train, y_train, X_test, y_test))

    df_metrics = pd.DataFrame(rows)
    df_metrics["scenario"] = scenario_name

    # Orden por PR-AUC (muy útil en desbalance)
    df_metrics = df_metrics.sort_values(by=["pr_auc", "recall_pos", "f1_pos"], ascending=False)

    print(f"\n=== Resultados: {scenario_name} ===")
    print(df_metrics)

    plot_comparison(df_metrics, title=f"Comparación de modelos ({scenario_name})")
    return df_metrics


def main():
    # Escenario 1: con todas las variables
    res_all = run_scenario("con_puntaje", drop_cols=None)

    # Escenario 2: sin 'puntaje' por posible leakage (correlación muy alta observada en EDA)
    res_no_score = run_scenario("sin_puntaje", drop_cols=["puntaje"])

    final = pd.concat([res_all, res_no_score], ignore_index=True)

    best = final.sort_values(by=["pr_auc", "recall_pos", "f1_pos"], ascending=False).head(1)
    print("\n✅ Mejor configuración (según PR-AUC + Recall/F1):")
    print(best)


if __name__ == "__main__":
    main()


    
    
    
    
    ### Análisis comparativo de modelos

#En el escenario sin la variable puntaje, los modelos presentan desempeño consistente con métricas realistas. RandomForest y XGBoost muestran recall cercano a 1.0 y F1 elevado, con ROC-AUC aproximado de 0.67.

#En el escenario con la variable puntaje, todos los modelos alcanzan métricas perfectas (1.0), lo cual sugiere posible fuga de información o una variable con capacidad predictiva casi determinística respecto al target.

#Dado que en el EDA se identificó una correlación de 0.92 entre puntaje y la variable objetivo, se considera que su inclusión puede comprometer la capacidad de generalización del modelo.

#Por este motivo, se selecciona como modelo final el RandomForest entrenado sin la variable puntaje, por presentar alto recall, F1 robusto y desempeño realista sin evidencia de sobreajuste extremo.