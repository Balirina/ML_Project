# 🚗 Car Price Prediction: Dual-Model Approach
Este proyecto desarrolla un sistema de predicción de precios de vehículos utilizando un dataset de **Kaggle.** El enfoque principal consiste en dividir el mercado en dos segmentos (Coches Económicos y Coches de Lujo) para mejorar la precisión de los modelos.

## 📋 Descripción del Proyecto
El objetivo es predecir el precio de venta de un vehículo basándose en características técnicas y de marca. Debido a la alta varianza en los precios, el dataset se dividió en dos subconjuntos:

**Coches baratos**: Precio <= 60,000.

**Coches caros**: Precio > 60,000.

Esta división permite que los modelos se especialicen en rangos de precios específicos, reduciendo el error en ambos extremos.

## 🛠️ Tecnologías Utilizadas
**Python 3.10+**

**Pandas & NumPy**: Manipulación y limpieza de datos.

**Matplotlib & Seaborn**: Análisis Exploratorio de Datos (EDA).

**Scikit-Learn**: Modelado y preprocesamiento.

**TensorFlow/Keras**: (Opcional, si usaste redes neuronales).

**Streamlit**: Interfaz de usuario (si llegas a crear la app).


## 📂 Estructura del Proyecto

├── app_streamlit/      # Aplicación principal (Streamlit)
├── data/               # Datasets originales y procesados
├── img/                # Las imagenes que se han usado
├── notebooks/          # Notebooks de Jupyter con el EDA y Entrenamiento
├── models/             # Modelos entrenados (.pkl)
├── src/                # Scripts en Python para ejecutar el proyecto
└── README.md           # Descripción del proyecto


## 🚀 Flujo Principal del trabjo

**Limpieza**: Tratamiento y eliminación de valores nulos.

**Feature Engineering**: Transformación de columnas categóricas (Brand, Model, Seller, Fuel, Type...) mediante distintos mapeos.

**EDA**: Análisis de correlaciones y detección de outliers.

**Segmentación**: División del dataset en el umbral de 60,000 para entrenamiento especializado.

**Entrenar modelos**: Probar diversos modelos. El criterio de selección fue el MAE (Mean Absolute Error).

**Analisis de las metricas**: Evaluación de las metricas.

**App de Streamlit**: Crear una interfaz para el usuario para que pueda introducir datos de coches y predecir el precio.

 ---

### Presentacíon en Prezi:

([link](https://prezi.com/p/edit/jpcdqftgtjji/))

---

## ✍️ Author

Proyecto elaborado por **Irina Balica**

