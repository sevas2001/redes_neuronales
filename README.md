# 🧬 Predicción de supervivencia en pacientes oncológicos con Deep Learning

Este repositorio contiene el código y los recursos del proyecto:



> **“Machine Learning para la predicción de supervivencia en pacientes oncológicos:  
> Modelado, evaluación y desarrollo de una herramienta clínica interactiva”**

Desarrollado como trabajo académico en la **Universidad Alfonso X el Sabio (UAX)**.

---

## 📌 Objetivo del proyecto

El objetivo principal es construir un sistema de apoyo a la decisión clínica capaz de:

1. **Predecir la probabilidad de supervivencia** de un paciente oncológico (vive / muere) a partir de un conjunto de variables clínicas, analíticas y anatomopatológicas.
2. **Comparar un modelo de deep learning (MLP en PyTorch)** frente a un modelo clásico de Machine Learning.
3. **Desplegar una aplicación web (Dash)** que permita a un médico introducir los datos de un paciente y obtener una estimación de riesgo de forma sencilla y visual.

---

## 🧠 Metodología

### 1. Preprocesado

- Separación de variables **numéricas** y **categóricas**.
- Tratamiento de valores faltantes:
  - Imputación por la **mediana** en numéricas.
  - Creación de indicadores `*_missing` para marcar ausencias.
  - Categoría especial `"MISSING"` en variables categóricas.
- Codificación de categóricas con `LabelEncoder`.
- Estandarización de numéricas con `StandardScaler`.
- División en **train / valid / test**.
- Conversión a tensores de PyTorch (con soporte para GPU cuando está disponible).

### 2. Modelos

- **Baseline**: modelo clásico de Machine Learning  
  (p. ej. Regresión logística o Random Forest) como referencia.
- **Modelo principal**: red neuronal multicapa (MLP) en PyTorch, con:
  - Varias capas densas.
  - Activaciones ReLU.
  - Dropout para regularización.
  - Entrenamiento con Adam, Binary Cross Entropy, early stopping y validación.

### 3. Evaluación

- Métricas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC (Average Precision)
- Gráficos:
  - Curva ROC
  - Curva Precision–Recall
  - Matriz de confusión
  - Curvas de pérdida (train vs valid)

---

## 💻 Aplicación web (Dash)

(La aplicación a sido subida a una web al que se puede acceder desde este enlace)
:  https://huggingface.co/spaces/sevas14/redes_neurnales_oncologia
El archivo `app.py` implementa una aplicación web con:

- Dentro la app se encuentran todos los pacientes, con el buscador encontraras el paceintes con las caracteristicas buscadas.
- Preprocesado interno que replica el pipeline usado en entrenamiento.
- Cálculo de la **probabilidad de supervivencia** usando el modelo MLP.
- Visualización del riesgo:
  - Probabilidad de supervivencia y de fallecimiento.
  - Barra de riesgo tipo “termómetro” con colores (verde/azul/naranja/rojo).
  - Mensaje interpretativo según nivel de riesgo.
- Aviso de que la herramienta es de apoyo y **no sustituye** el criterio médico.

---

