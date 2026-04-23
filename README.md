# Pipeline MLOps para Riesgo Crediticio

Proyecto end-to-end de Machine Learning y MLOps para predecir probabilidad de pago a tiempo, incluyendo entrenamiento, API, monitoreo de drift y dashboard de observabilidad.

---

## Descripción del proyecto

Este proyecto busca simular un flujo productivo de MLOps aplicado a riesgo crediticio.

Incluye:

- Ingeniería de variables
- Benchmarking de modelos
- Detección de leakage
- Pipeline final serializado para producción
- API de inferencia con FastAPI
- Monitoreo de data drift
- Dashboard interactivo de observabilidad
- Ejecución dockerizada

---

## Problema de negocio

El objetivo es estimar si un cliente tiene probabilidad de **pagar a tiempo** usando variables financieras, demográficas y crediticias.

Además del modelado, el foco está en llevar una solución desde experimentación hacia una arquitectura más cercana a producción:

- seleccionar un modelo robusto
- evitar leakage
- exponer predicciones por API
- monitorear drift en el tiempo

---

## Componentes principales

### Modelado
Se compararon:

- Logistic Regression
- Random Forest
- XGBoost

Se evaluaron dos escenarios:

### Escenario con `puntaje`
Obtuvo métricas casi perfectas y permitió detectar leakage.

### Escenario sin `puntaje`
Produjo desempeño más realista y fue elegido para deploy.

---

## MLOps y Serving

El pipeline final se serializa como:

```bash
models/model_pipeline.joblib
```

Se expone mediante FastAPI con endpoints:

- `/`
- `/health`
- `/predict`
- `/docs`

Ejecutar:

```bash
uvicorn src.model_deploy:app --reload
```

---

## Monitoreo de Drift

Se implementa monitoreo usando:

- PSI
- Jensen-Shannon
- KS test
- Chi-cuadrado

Archivos generados:

- baseline_reference.csv
- drift_report.csv
- drift_history.csv

Ejecutar:

```bash
python -m src.model_monitoring
```

---

## Dashboard de Observabilidad

Aplicación en Streamlit para monitoreo interactivo:

```bash
streamlit run src/app.py
```

Incluye:

- Primary incident detection
- Global drift risk gauge
- Drift analysis table
- Top drifted features
- Recommended actions
- Feature diagnostics
- Historical monitoring
- Feature importance vs drift matrix

---

## Stack Tecnológico

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Plotly
- Docker
- Render

---

## Estructura

```bash
mlops_pipeline/
│
├── models/
│   └── model_pipeline.joblib
│
├── src/
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── app.py
│
├── baseline_reference.csv
├── drift_report.csv
├── drift_history.csv
├── feature_importance.csv
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Docker

Levantar todo:

```bash
docker compose up --build
```

Servicios:

- API → localhost:8000
- Swagger → localhost:8000/docs
- Dashboard → localhost:8501

---

## Próximos pasos

Posibles mejoras futuras:

- Prediction drift
- Retraining triggers
- Alertas automáticas
- Champion vs Challenger monitoring
- Explainability drift

---

## Autora

**Florencia Sosa Comisso**  
Data & Business Analyst | BI · SQL · Python · Power BI

LinkedIn:  
http://linkedin.com/in/florencia-sosa-comisso
