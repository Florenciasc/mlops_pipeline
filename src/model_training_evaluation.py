from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

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

from src.ft_engineering import prepare_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_pipeline.joblib"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "feature_importance.csv"


def summarize_classification(y_true, y_pred, y_proba) -> dict:
    """Métricas principales para dataset desbalanceado."""
    return {
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def build_model(model, preprocessor) -> Pipeline:
    """Pipeline completo: preprocess + modelo."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_eval_one(
    name: str,
    model,
    preprocessor,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    pipe = build_model(model, preprocessor)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = summarize_classification(y_test, y_pred, y_proba)
    metrics["model"] = name
    return metrics


def plot_comparison(df_metrics, title):
    """Se deja como placeholder por si luego querés agregar visualización."""
    return None


def get_models() -> dict:
    return {
        "LogReg_balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        ),
    }


def run_scenario(scenario_name: str, drop_cols=None) -> pd.DataFrame:
    TARGET = "pago_atiempo"

    split, preprocessor, feature_names = prepare_training_data(
        target=TARGET,
        drop_cols=drop_cols,
    )

    X_train, X_test = split.X_train, split.X_test
    y_train, y_test = split.y_train, split.y_test

    models = get_models()

    rows = []
    for name, model in models.items():
        rows.append(
            train_eval_one(
                name=name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        )

    df_metrics = pd.DataFrame(rows)
    df_metrics["scenario"] = scenario_name

    df_metrics = df_metrics.sort_values(
        by=["pr_auc", "recall_pos", "f1_pos"],
        ascending=False,
    )

    print(f"\n=== Resultados: {scenario_name} ===")
    print(df_metrics)

    return df_metrics


def train_final_pipeline(best_model_name: str, drop_cols=None):
    TARGET = "pago_atiempo"

    split, preprocessor, feature_names = prepare_training_data(
        target=TARGET,
        drop_cols=drop_cols,
    )

    models = get_models()
    final_model = models[best_model_name]

    final_pipeline = build_model(final_model, preprocessor)
    final_pipeline.fit(split.X_train, split.y_train)

    return final_pipeline, final_model, preprocessor


def export_feature_importance(final_model, preprocessor) -> None:
    if not hasattr(final_model, "feature_importances_"):
        print("ℹ️ El modelo final no expone feature_importances_. No se genera feature_importance.csv")
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = final_model.feature_importances_

    cleaned_names = []
    for col in feature_names:
        col = str(col)
        col = col.replace("num__", "").replace("cat__", "")
        if "__" in col:
            col = col.split("__")[-1]
        cleaned_names.append(col)

    df_importance = pd.DataFrame(
        {
            "variable": cleaned_names,
            "importance": importances,
        }
    )

    df_importance = (
        df_importance.groupby("variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    df_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print("✅ Feature importance guardada en feature_importance.csv")


def main():
    # Escenario 1: con puntaje
    res_all = run_scenario(
        scenario_name="con_puntaje",
        drop_cols=None,
    )

    # Escenario 2: sin puntaje
    res_no_score = run_scenario(
        scenario_name="sin_puntaje",
        drop_cols=["puntaje"],
    )

    final = pd.concat(
        [res_all, res_no_score],
        ignore_index=True,
    )

    best = final.sort_values(
        by=["pr_auc", "recall_pos", "f1_pos"],
        ascending=False,
    ).head(1)

    print("\n✅ Mejor configuración:")
    print(best)

    best_model_name = best.iloc[0]["model"]
    best_scenario = best.iloc[0]["scenario"]

    best_drop_cols = None if best_scenario == "con_puntaje" else ["puntaje"]

    print("\nEntrenando pipeline final para deploy...")

    final_pipeline, final_model, preprocessor = train_final_pipeline(
        best_model_name=best_model_name,
        drop_cols=best_drop_cols,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    export_feature_importance(final_model, preprocessor)

    print("✅ Modelo guardado en models/model_pipeline.joblib")


if __name__ == "__main__":
    main()

