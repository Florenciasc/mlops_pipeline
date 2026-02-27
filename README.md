# Proyecto Integrador - Modelo Predictivo de Crédito

## Objetivo
Desarrollar un modelo supervisado para predecir si un cliente realizará su pago a tiempo.

## Dataset
Base histórica de créditos con variable objetivo `Pago_atiempo`.

## Avance 1
- Estructura de proyecto definida
- Carga y limpieza inicial de datos
- Análisis Exploratorio completo (univariable, bivariable, multivariable)

## Observaciones
Se detectó desbalance en la variable objetivo, el cual será tratado en etapas posteriores.

## Resultados del EDA

- Se detectó desbalance en la variable objetivo (~95/5).
- Se identificaron variables altamente correlacionadas con el target.
- Se observaron distribuciones asimétricas en variables monetarias.
- Se definieron reglas preliminares de validación de datos.

## Avance 2
Ingeniería de características

Se normalizaron nombres de columnas y se unificaron valores nulos.

Se extrajeron variables temporales (anio_prestamo, mes_prestamo) a partir de la fecha.

Se trató tipo_credito como variable categórica codificada.

Se construyó un pipeline reproducible con imputación, escalado y OneHotEncoding.

Modelamiento

Se entrenaron Logistic Regression, Random Forest y XGBoost.

Se priorizaron métricas robustas para desbalance (PR-AUC, Recall, F1).

Se evaluaron dos escenarios: con y sin la variable puntaje.

Hallazgos clave

La variable puntaje mostró correlación muy alta con el target (0.92).

El modelo XGBoost con puntaje obtuvo performance perfecta (1.0 en todas las métricas).

Este resultado sugiere posible fuga de información.

Modelo seleccionado

Se selecciona RandomForest sin la variable puntaje por:

Alto recall (1.0)

F1 elevado

Ausencia de indicios de sobreajuste extremo

Mayor robustez y potencial de generalización