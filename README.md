# End-to-End MLOps Pipeline for Credit Risk Monitoring

Production-style MLOps pipeline for credit risk prediction, model serving, drift detection and observability.

## Overview

This project simulates an end-to-end machine learning system in a production environment, covering the full lifecycle from model training to deployment and monitoring.

It includes:

- Predictive model for credit behavior
- FastAPI model serving API
- Dockerized deployment
- Data drift monitoring framework
- Interactive Streamlit observability dashboard
- Deployment in Render (API + Monitoring Console)

The project was designed following MLOps principles: reproducibility, monitoring, model governance and operational visibility.

---

## Architecture

```text
Data → Feature Engineering → Model Training → API Inference
                                     ↓
                           Drift Monitoring Layer
                                     ↓
                           Monitoring Dashboard
```

### Components

### 1. Model Training
- Feature engineering pipeline
- Classification model for `Pago_atiempo`
- Serialized production pipeline with Joblib

Algorithms explored:
- Logistic Regression
- Gradient Boosting approaches
- Risk-oriented evaluation metrics

---

## 2. Prediction API (FastAPI)

Deployed API:

https://mlops-pipeline-api.onrender.com

Endpoints:

### Health Check
```http
GET /
```

### Batch Predictions
```http
POST /predict
```

Returns:

- predicted class
- probability score
- model version
- threshold metadata

Example response:

```json
{
 "predictions":[
   {
    "proba_pago_atiempo":0.983,
    "pred_pago_atiempo":1
   }
 ]
}
```

Swagger Docs:

```http
https://mlops-pipeline-api.onrender.com/docs
```

---

## 3. Drift Monitoring Console

Live dashboard:

https://mlops-pipeline-dashboard.onrender.com

Monitoring includes:

Numerical drift:
- PSI
- Kolmogorov-Smirnov
- Jensen-Shannon Distance

Categorical drift:
- Chi-Square Drift Detection

Dashboard modules:

- Executive Risk Overview
- Drift Incident Panel
- Feature Diagnostics
- Historical Drift Monitoring
- Risk Scoring Matrix
- Downloadable Monitoring Reports

---

## Monitoring Example

Detected incident:

- Feature: `tendencia_ingresos`
- Status: DRIFT
- Automatic recommendation:
  Review upstream category mapping

Simulates production monitoring workflows.

---

## Tech Stack

Python  
Pandas  
Scikit-learn  
FastAPI  
Streamlit  
Docker  
Render  
Joblib  
Plotly

---

## Project Structure

```bash
mlops_pipeline/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── app.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── ft_engineering.py
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Local Run

Clone repository:

```bash
git clone https://github.com/Florenciasc/mlops_pipeline.git
cd mlops_pipeline
```

Install:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn src.model_deploy:app --reload
```

Run monitoring dashboard:

```bash
streamlit run src/app.py
```

---

## Key MLOps Capabilities Demonstrated

✔ Model deployment  
✔ Batch inference API  
✔ Data drift detection  
✔ Model observability  
✔ Incident-style monitoring  
✔ Dockerized deployment  
✔ Cloud deployment  
✔ Monitoring dashboard

---

## Future Improvements

- Model performance monitoring (prediction drift)
- Automated alerting
- CI/CD pipeline integration
- Retraining triggers
- Prometheus/Grafana integration

---

## Author

Florencia Sosa Comisso  
Business & Data Analyst | MLOps | BI

LinkedIn:
https://linkedin.com/in/florencia-sosa-comisso

Portfolio:
https://florenciasc.github.io/
