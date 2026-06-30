import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Inclusión Financiera Municipal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS PERSONALIZADOS
# ============================================================

def aplicar_estilos():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7fb;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            padding: 2rem;
            border-radius: 18px;
            color: white;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .hero-text {
            font-size: 1rem;
            line-height: 1.6;
            opacity: 0.95;
        }

        .info-card {
            background: white;
            padding: 1.2rem;
            border-radius: 16px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            border-left: 6px solid #2563eb;
        }

        .info-card h4 {
            margin: 0;
            color: #0f172a;
            font-size: 1rem;
        }

        .info-card p {
            margin: 0.35rem 0 0 0;
            color: #475569;
            font-size: 0.95rem;
        }

        .result-card {
            background: white;
            padding: 1.5rem;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.10);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.8rem;
        }

        .small-note {
            font-size: 0.9rem;
            color: #64748b;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e0ecff, #f8fbff);
        }

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #0f172a;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-weight: 600;
            width: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

aplicar_estilos()

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
# FUNCIONES DE APOYO
# ============================================================

def calcular_indice(datos_usuario):
    datos_indice = pd.DataFrame([datos_usuario])[variables_indice]
    datos_indice_log = np.log1p(datos_indice)
    datos_indice_normalizados = scaler_indice.transform(datos_indice_log)
    indice = datos_indice_normalizados.mean(axis=1)[0]
    return indice


def predecir_nivel(datos_usuario):
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
# ENCABEZADO PRINCIPAL
# ============================================================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🏦 Predicción del Nivel de Inclusión Financiera Municipal</div>
    <div class="hero-text">
        Esta aplicación estima si un municipio puede presentar un nivel de inclusión financiera 
        <b>bajo, medio o alto</b> en el siguiente corte de información.  
        El modelo utiliza variables asociadas a corresponsales, operaciones financieras, 
        cuentas de ahorro, crédito de consumo, microcrédito y presencia de entidades financieras.
    </div>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    <div class="info-card">
        <h4>🎯 Objetivo</h4>
        <p>Apoyar el análisis de cobertura e inclusión financiera municipal mediante inteligencia artificial.</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="info-card">
        <h4>🧠 Modelo</h4>
        <p>Red neuronal supervisada entrenada con información financiera municipal de Colombia.</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class="info-card">
        <h4>📌 Resultado</h4>
        <p>Clasificación estimada del municipio en nivel <b>bajo</b>, <b>medio</b> o <b>alto</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR - FORMULARIO
# ============================================================

with st.sidebar:
    st.markdown("## 📥 Ingreso de datos")
    st.markdown("Completa la información financiera del municipio para generar la predicción.")

    with st.form("formulario_prediccion"):
        anio = st.number_input("Año", min_value=2017, max_value=2030, value=2021, step=1)
        nro_corresponsales = st.number_input("Número de corresponsales", min_value=0, value=10, step=1)
        nro_total = st.number_input("Número total de operaciones", min_value=0.0, value=1000.0, step=100.0)
        monto_total = st.number_input("Monto total de operaciones", min_value=0.0, value=50000000.0, step=1000000.0)

        nro_total_cta_ahorros = st.number_input("Número total de cuentas de ahorro", min_value=0.0, value=500.0, step=50.0)
        saldo_total_cta_ahorros = st.number_input("Saldo total de cuentas de ahorro", min_value=0.0, value=100000000.0, step=1000000.0)

        nro_total_credito_consumo = st.number_input("Número total de créditos de consumo", min_value=0, value=100, step=10)
        monto_total_credito_consumo = st.number_input("Monto total de crédito de consumo", min_value=0.0, value=20000000.0, step=1000000.0)

        nro_total_microcredito = st.number_input("Número total de microcréditos", min_value=0, value=20, step=5)
        monto_total_microcredito = st.number_input("Monto total de microcrédito", min_value=0.0, value=10000000.0, step=500000.0)

        cantidad_entidades = st.number_input("Cantidad de entidades financieras", min_value=0, value=3, step=1)
        cantidad_tipos_producto = st.number_input("Cantidad de tipos de producto", min_value=0, value=3, step=1)

        boton = st.form_submit_button("🔎 Predecir nivel de inclusión financiera")

    st.markdown("---")
    st.markdown("### ℹ️ Nota")
    st.markdown(
        '<p class="small-note">Esta herramienta corresponde a un ejercicio académico y no representa una medición oficial.</p>',
        unsafe_allow_html=True
    )

# ============================================================
# RESULTADO
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

    st.markdown('<div class="section-title">Resultado de la predicción</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Nivel estimado", nivel_predicho.upper())
            st.metric("Índice calculado", round(indice, 4))
            st.markdown('</div>', unsafe_allow_html=True)

            if nivel_predicho == "alto":
                st.success("El municipio presenta señales asociadas a un nivel alto de inclusión financiera.")
            elif nivel_predicho == "medio":
                st.warning("El municipio presenta señales asociadas a un nivel medio de inclusión financiera.")
            else:
                st.error("El municipio presenta señales asociadas a un nivel bajo de inclusión financiera.")

    with c2:
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### Probabilidades por categoría")
            st.dataframe(tabla_probabilidades, use_container_width=True)
            st.bar_chart(tabla_probabilidades.set_index("Nivel")["Probabilidad (%)"])
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-title">Vista previa del sistema</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        st.markdown("""
        <div class="info-card">
            <h4>¿Cómo funciona?</h4>
            <p>
                El usuario ingresa variables financieras del municipio. Luego, la aplicación calcula
                un índice de inclusión financiera, aplica las mismas transformaciones del entrenamiento
                y genera una predicción con el modelo de inteligencia artificial.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="info-card">
            <h4>¿Qué muestra el sistema?</h4>
            <p>
                El sistema presenta el nivel estimado de inclusión financiera, el índice calculado y
                las probabilidades asociadas a cada categoría: bajo, medio y alto.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("---")
st.caption(
    "Proyecto académico de Deep Learning | Predicción del nivel de inclusión financiera municipal en Colombia"
)
