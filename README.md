# 🚀 MLOps Pipeline - Customer Churn Prediction

Proyecto de Machine Learning orientado a la predicción de churn de clientes, desarrollado bajo buenas prácticas de MLOps para garantizar reproducibilidad, organización y despliegue.

---

## 🎯 Objetivo

Construir un pipeline completo que permita:

* Analizar datos de clientes
* Entrenar modelos de clasificación
* Evaluar su performance
* Servir predicciones mediante una API
* Preparar el sistema para despliegue en producción

---

## 🧠 Enfoque

Este proyecto va más allá del modelo.

👉 Se enfoca en el ciclo completo:

**Datos → Modelado → API → Docker → Deploy**

---

## 📁 Estructura del proyecto

```bash
mlops_pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model_pipeline.joblib
│
├── notebooks/
│   ├── cargar_datos.ipynb
│   └── comprension_eda.ipynb
│
├── reports/
│
├── src/
│   ├── app.py                  # API (FastAPI)
│   ├── train.py               # Entrenamiento
│   ├── ft_engineering.py      # Feature engineering
│   ├── model_deploy.py        # Lógica de predicción
│   ├── model_monitoring.py    # Monitoreo
│   └── model_training_evaluation.py
│
├── tests/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

```bash
git clone https://github.com/Florenciasc/mlops_pipeline.git
cd mlops_pipeline
```

Crear entorno virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## 🧪 Entrenamiento del modelo

```bash
python src/train.py
```

El modelo entrenado se guarda en:

```
models/model_pipeline.joblib
```

---

## 🌐 Levantar API

```bash
uvicorn src.app:app --reload
```

Disponible en:

```
http://localhost:8000
```

---

## 🔍 Ejemplo de uso

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"feature1": value1, "feature2": value2}'
```

---

## 🐳 Docker

Construir imagen:

```bash
docker build -t churn-api .
```

Correr contenedor:

```bash
docker run -p 8000:8000 churn-api
```

---

## ☁️ Deploy

El proyecto está preparado para deploy en cloud (Render, AWS, GCP, Azure).

👉 Próximo paso:

* deploy en producción
* automatización (CI/CD)

---

## 📊 Métricas evaluadas

* Accuracy
* Precision
* Recall
* ROC-AUC

---

## 🚀 Roadmap

* [ ] Deploy en la nube
* [ ] Pipeline automático
* [ ] Monitoreo en producción
* [ ] Versionado de modelos

---

## 👩‍💻 Autor

Florencia Sosa Comisso
🔗 http://linkedin.com/in/florencia-sosa-comisso
