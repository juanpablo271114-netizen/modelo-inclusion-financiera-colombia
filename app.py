import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ============================================================
# CONFIGURACIÓN GENERAL DE LA APLICACIÓN
# ============================================================

st.set_page_config(
    page_title="Modelo de Inclusión Financiera",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# CARGA DE MODELO Y OBJETOS GUARDADOS
# ============================================================

@st.cache_resource
def cargar_recursos():
    modelo = tf.keras.models.load_model("modelo_guardado/modelo_inclusion_financiera.keras")
    scaler_modelo = joblib.load("modelo_guardado/scaler_modelo.pkl")
    scaler_indice = joblib.load("modelo_guardado/scaler_indice.pkl")
    encoder = joblib.load("modelo_guardado/encoder_niveles.pkl")
    variables_entrada = joblib.load("modelo_guardado/variables_entrada.pkl")
    variables_indice = joblib.load("modelo_guardado/variables_indice.pkl")

    return modelo, scaler_modelo, scaler_indice, encoder, variables_entrada, variables_indice


modelo, scaler_modelo, scaler_indice, encoder, variables_entrada, variables_indice = cargar_recursos()

# ============================================================
# ENCABEZADO
# ============================================================

st.title("🏦 Predicción del Nivel de Inclusión Financiera Municipal")

st.markdown(
    """
    Esta aplicación permite estimar si un municipio puede presentar un nivel de inclusión financiera
    **bajo, medio o alto** en el siguiente corte de información.

    El modelo utiliza variables relacionadas con corresponsales, operaciones financieras,
    cuentas de ahorro, crédito de consumo, microcrédito y presencia de entidades financieras.
    """
)

st.info(
    """
    Esta herramienta corresponde a un ejercicio académico. La predicción se basa en un índice
    construido a partir de variables financieras disponibles en el dataset y no representa
    una medición oficial de inclusión financiera.
    """
)

# ============================================================
# FORMULARIO DE ENTRADA
# ============================================================

st.subheader("Ingrese los datos financieros del municipio")

with st.form("formulario_prediccion"):

    col1, col2, col3 = st.columns(3)

    with col1:
        anio = st.number_input("Año", min_value=2017, max_value=2030, value=2021, step=1)
        nro_corresponsales = st.number_input("Número de corresponsales", min_value=0, value=10, step=1)
        nro_total = st.number_input("Número total de operaciones", min_value=0.0, value=1000.0, step=100.0)
        monto_total = st.number_input("Monto total de operaciones", min_value=0.0, value=50000000.0, step=1000000.0)

    with col2:
        nro_total_cta_ahorros = st.number_input("Número total de cuentas de ahorro", min_value=0.0, value=500.0, step=50.0)
        saldo_total_cta_ahorros = st.number_input("Saldo total de cuentas de ahorro", min_value=0.0, value=100000000.0, step=1000000.0)
        nro_total_credito_consumo = st.number_input("Número total de créditos de consumo", min_value=0, value=100, step=10)
        monto_total_credito_consumo = st.number_input("Monto total de crédito de consumo", min_value=0.0, value=20000000.0, step=1000000.0)

    with col3:
        nro_total_microcredito = st.number_input("Número total de microcréditos", min_value=0, value=20, step=5)
        monto_total_microcredito = st.number_input("Monto total de microcrédito", min_value=0.0, value=10000000.0, step=500000.0)
        cantidad_entidades = st.number_input("Cantidad de entidades financieras", min_value=0, value=3, step=1)
        cantidad_tipos_producto = st.number_input("Cantidad de tipos de producto", min_value=0, value=3, step=1)

    boton = st.form_submit_button("Predecir nivel de inclusión financiera")

# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def calcular_indice(datos_usuario):
    """
    Calcula el índice de inclusión financiera usando la misma lógica aplicada
    durante el entrenamiento del modelo.
    """

    datos_indice = pd.DataFrame([datos_usuario])[variables_indice]
    datos_indice_log = np.log1p(datos_indice)
    datos_indice_normalizados = scaler_indice.transform(datos_indice_log)

    indice = datos_indice_normalizados.mean(axis=1)[0]

    return indice


def predecir_nivel(datos_usuario):
    """
    Prepara los datos ingresados por el usuario y genera la predicción.
    """

    datos_usuario["indice_inclusion_financiera"] = calcular_indice(datos_usuario)

    datos = pd.DataFrame([datos_usuario])
    datos = datos[variables_entrada]

    datos_log = np.log1p(datos)
    datos_scaled = scaler_modelo.transform(datos_log)

    prediccion_prob = modelo.predict(datos_scaled, verbose=0)[0]
    prediccion_num = np.argmax(prediccion_prob)
    prediccion_texto = encoder.inverse_transform([prediccion_num])[0]

    tabla_probabilidades = pd.DataFrame({
        "Nivel": encoder.classes_,
        "Probabilidad (%)": (prediccion_prob * 100).round(2)
    })

    return prediccion_texto, tabla_probabilidades, datos_usuario["indice_inclusion_financiera"]

# ============================================================
# RESULTADO DE LA PREDICCIÓN
# ============================================================

if boton:

    datos_usuario = {
        "anio": anio,
        "nro_corresponsales": nro_corresponsales,
        "nro_total": nro_total,
        "monto_total": monto_total,
        "nro_total_cta_ahorros": nro_total_cta_ahorros,
        "saldo_total_cta_ahorros": saldo_total_cta_ahorros,
        "nro_total_credito_consumo": nro_total_credito_consumo,
        "monto_total_credito_consumo": monto_total_credito_consumo,
        "nro_total_microcredito": nro_total_microcredito,
        "monto_total_microcredito": monto_total_microcredito,
        "cantidad_entidades": cantidad_entidades,
        "cantidad_tipos_producto": cantidad_tipos_producto
    }

    nivel_predicho, tabla_probabilidades, indice = predecir_nivel(datos_usuario)

    st.markdown("---")
    st.subheader("Resultado de la predicción")

    col_resultado1, col_resultado2 = st.columns([1, 2])

    with col_resultado1:
        st.metric("Nivel estimado", nivel_predicho.upper())
        st.metric("Índice calculado", round(indice, 4))

    with col_resultado2:
        st.write("Probabilidades por categoría")
        st.dataframe(tabla_probabilidades, use_container_width=True)
        st.bar_chart(tabla_probabilidades.set_index("Nivel")["Probabilidad (%)"])

    if nivel_predicho == "alto":
        st.success("El municipio presenta señales asociadas a un nivel alto de inclusión financiera.")
    elif nivel_predicho == "medio":
        st.warning("El municipio presenta señales asociadas a un nivel medio de inclusión financiera.")
    else:
        st.error("El municipio presenta señales asociadas a un nivel bajo de inclusión financiera.")
