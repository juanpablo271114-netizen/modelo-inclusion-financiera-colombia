# Modelo de Inclusión Financiera Municipal en Colombia

Este proyecto corresponde a un ejercicio académico de Deep Learning aplicado al análisis de inclusión financiera municipal en Colombia.

## Objetivo

Construir un modelo de inteligencia artificial que estime si un municipio podría presentar un nivel de inclusión financiera **bajo**, **medio** o **alto** en el siguiente corte de información, a partir de variables relacionadas con:

- corresponsales financieros;
- número y monto de operaciones;
- cuentas de ahorro;
- crédito de consumo;
- microcrédito;
- cantidad de entidades financieras;
- variedad de tipos de producto.

## Estructura del proyecto

```text
modelo_inclusion_financiera_colombia/
├── app.py
├── requirements.txt
├── .gitignore
└── modelo_guardado/
    ├── modelo_inclusion_financiera.keras
    ├── scaler_modelo.pkl
    ├── scaler_indice.pkl
    ├── encoder_niveles.pkl
    ├── variables_entrada.pkl
    ├── variables_indice.pkl
    └── base_modelo_inclusion_financiera.csv
```

## Cómo ejecutar la aplicación

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecutar la aplicación:

```bash
streamlit run app.py
```

3. Ingresar los datos financieros del municipio y presionar el botón **Predecir nivel de inclusión financiera**.

## Resultado del modelo

El modelo clasifica el nivel estimado de inclusión financiera en tres categorías:

- bajo;
- medio;
- alto.

También muestra las probabilidades estimadas para cada categoría.

## Nota académica

La predicción corresponde a una aproximación académica basada en un índice construido a partir de las variables disponibles en el dataset. No representa una medición oficial de inclusión financiera.
