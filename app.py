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
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #f4f7fb;
                --primary: #0f172a;
                --secondary: #1d4ed8;
                --accent: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --muted: #64748b;
                --card: #ffffff;
                --border: #e2e8f0;
            }

            .main {
                background-color: var(--bg-main);
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1350px;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            }

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] div {
                color: #f8fafc;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #ffffff;
            }

            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #10b981 100%);
                padding: 2.2rem;
                border-radius: 24px;
                color: white;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.25);
                margin-bottom: 1.5rem;
            }

            .hero h1 {
                font-size: 2.25rem;
                margin-bottom: 0.5rem;
                font-weight: 800;
                letter-spacing: -0.03em;
            }

            .hero p {
                font-size: 1.02rem;
                line-height: 1.65;
                max-width: 980px;
                opacity: 0.96;
            }

            .badge {
                display: inline-block;
                background: rgba(255,255,255,0.16);
                border: 1px solid rgba(255,255,255,0.30);
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.85rem;
                margin-bottom: 0.8rem;
                font-weight: 600;
            }

            .card {
                background: var(--card);
                border: 1px solid var(--border);
                padding: 1.25rem;
                border-radius: 20px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
                height: 100%;
            }

            .card-title {
                color: #0f172a;
                font-weight: 800;
                font-size: 1.05rem;
                margin-bottom: 0.45rem;
            }

            .card-text {
                color: #475569;
                font-size: 0.94rem;
                line-height: 1.55;
            }

            .section-title {
                color: #0f172a;
                font-size: 1.35rem;
                font-weight: 800;
                margin-top: 0.5rem;
                margin-bottom: 0.9rem;
                letter-spacing: -0.02em;
            }

            .result-panel {
                background: white;
                border-radius: 24px;
                padding: 1.5rem;
                box-shadow: 0 16px 38px rgba(15, 23, 42, 0.10);
                border: 1px solid #e2e8f0;
                margin-top: 1rem;
            }

            .result-high {
                background: linear-gradient(135deg, #ecfdf5, #d1fae5);
                border: 1px solid #10b981;
                color: #065f46;
                padding: 1.2rem;
                border-radius: 20px;
                font-weight: 800;
                font-size: 1.4rem;
                text-align: center;
            }

            .result-medium {
                background: linear-gradient(135deg, #fffbeb, #fef3c7);
                border: 1px solid #f59e0b;
                color: #92400e;
                padding: 1.2rem;
                border-radius: 20px;
                font-weight: 800;
                font-size: 1.4rem;
                text-align: center;
            }

            .result-low {
                background: linear-gradient(135deg, #fef2f2, #fee2e2);
                border: 1px solid #ef4444;
                color: #991b1b;
                padding: 1.2rem;
                border-radius: 20px;
                font-weight: 800;
                font-size: 1.4rem;
                text-align: center;
            }

            .metric-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                padding: 1rem;
                border-radius: 18px;
                text-align: center;
            }

            .metric-label {
                color: #64748b;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
            }

            .metric-value {
                color: #0f172a;
                font-size: 1.35rem;
                font-weight: 800;
            }

            .prob-row {
                margin-bottom: 1rem;
            }

            .prob-label {
                display: flex;
                justify-content: space-between;
                font-size: 0.92rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.35rem;
            }

            .prob-track {
                width: 100%;
                height: 14px;
                background: #e5e7eb;
                border-radius: 999px;
                overflow: hidden;
            }

            .prob-fill {
                height: 14px;
                border-radius: 999px;
            }

            .interpretation {
                background: #f8fafc;
                border-left: 6px solid #2563eb;
                padding: 1rem 1.2rem;
                border-radius: 16px;
                color: #334155;
                line-height: 1.55;
                font-size: 0.95rem;
            }

            .footer {
                text-align: center;
                color: #64748b;
                font-size: 0.85rem;
                margin-top: 2rem;
            }

            .stButton > button {
                background: linear-gradient(135deg, #2563eb, #1d4ed8);
                color: white;
                border: 0;
                border-radius: 14px;
                padding: 0.8rem 1rem;
                font-weight: 800;
                width: 100%;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.25);
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #1d4ed8, #1e40af);
                color: white;
                border: 0;
            }

            .stForm {
                border: 0;
            }

            hr {
                border: none;
                border-top: 1px solid #e2e8f0;
                margin-top: 1.5rem;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

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
# FUNCIONES
# ============================================================

def formato_numero(valor):
    return f"{valor:,.0f}".replace(",", ".")


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
        "Probabilidad": prediccion_prob
    })

    tabla_probabilidades["Probabilidad (%)"] = (
        tabla_probabilidades["Probabilidad"] * 100
    ).round(2)

    return prediccion_texto, tabla_probabilidades, datos_usuario["indice_inclusion_financiera"]


def bloque_probabilidades(tabla):
    colores = {
        "alto": "#10b981",
        "medio": "#f59e0b",
        "bajo": "#ef4444"
    }

    orden = ["alto", "medio", "bajo"]
    tabla = tabla.set_index("Nivel").loc[[nivel for nivel in orden if nivel in tabla["Nivel"].values]].reset_index()

    html = ""

    for _, fila in tabla.iterrows():
        nivel = fila["Nivel"]
        prob = fila["Probabilidad (%)"]
        color = colores.get(nivel, "#2563eb")

        html += f"""
        <div class="prob-row">
            <div class="prob-label">
                <span>{nivel.upper()}</span>
                <span>{prob:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{prob}%; background:{color};"></div>
            </div>
        </div>
        """

    return html


def interpretar_resultado(nivel, prob_max):
    if nivel == "alto":
        return (
            f"El modelo identifica señales fuertes de inclusión financiera. "
            f"La probabilidad más alta fue de {prob_max:.2f}%, lo que indica que las variables ingresadas "
            f"se parecen a municipios con mayor presencia de productos, operaciones y entidades financieras."
        )
    elif nivel == "medio":
        return (
            f"El modelo ubica el caso en un nivel intermedio. "
            f"La probabilidad más alta fue de {prob_max:.2f}%, lo que sugiere una condición financiera moderada, "
            f"con señales que no corresponden claramente a los extremos bajo o alto."
        )
    else:
        return (
            f"El modelo identifica señales asociadas a baja inclusión financiera. "
            f"La probabilidad más alta fue de {prob_max:.2f}%, lo que puede reflejar menor actividad financiera, "
            f"menor presencia de entidades o menor diversidad de productos."
        )


# ============================================================
# ESCENARIOS PREDEFINIDOS
# ============================================================

escenarios = {
    "Municipio pequeño con baja actividad": {
        "anio": 2021,
        "nro_corresponsales": 1,
        "nro_total": 80.0,
        "monto_total": 2500000.0,
        "nro_total_cta_ahorros": 30.0,
        "saldo_total_cta_ahorros": 12000000.0,
        "nro_total_credito_consumo": 2,
        "monto_total_credito_consumo": 3000000.0,
        "nro_total_microcredito": 1,
        "monto_total_microcredito": 1500000.0,
        "cantidad_entidades": 1,
        "cantidad_tipos_producto": 2
    },
    "Municipio intermedio": {
        "anio": 2021,
        "nro_corresponsales": 10,
        "nro_total": 1000.0,
        "monto_total": 50000000.0,
        "nro_total_cta_ahorros": 500.0,
        "saldo_total_cta_ahorros": 100000000.0,
        "nro_total_credito_consumo": 100,
        "monto_total_credito_consumo": 20000000.0,
        "nro_total_microcredito": 20,
        "monto_total_microcredito": 10000000.0,
        "cantidad_entidades": 3,
        "cantidad_tipos_producto": 3
    },
    "Municipio con alta actividad financiera": {
        "anio": 2021,
        "nro_corresponsales": 250,
        "nro_total": 850000.0,
        "monto_total": 85000000000.0,
        "nro_total_cta_ahorros": 180000.0,
        "saldo_total_cta_ahorros": 250000000000.0,
        "nro_total_credito_consumo": 35000,
        "monto_total_credito_consumo": 120000000000.0,
        "nro_total_microcredito": 1500,
        "monto_total_microcredito": 30000000000.0,
        "cantidad_entidades": 18,
        "cantidad_tipos_producto": 8
    }
}

# ============================================================
# ENCABEZADO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">Proyecto académico de Deep Learning aplicado</div>
        <h1>Predicción del Nivel de Inclusión Financiera Municipal</h1>
        <p>
            Sistema inteligente para estimar si un municipio colombiano podría presentar un nivel 
            <b>bajo</b>, <b>medio</b> o <b>alto</b> de inclusión financiera en el siguiente corte de información. 
            El modelo utiliza variables relacionadas con corresponsales, operaciones, cuentas de ahorro, 
            crédito de consumo, microcrédito y diversidad de entidades financieras.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TARJETAS INFORMATIVAS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Objetivo</div>
            <div class="card-text">
                Apoyar el análisis territorial de cobertura e inclusión financiera.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Modelo</div>
            <div class="card-text">
                Red neuronal simple entrenada con datos financieros municipales.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Predicción</div>
            <div class="card-text">
                Clasificación futura del municipio en nivel bajo, medio o alto.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Uso</div>
            <div class="card-text">
                El usuario ingresa variables y la app genera la estimación.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Panel de entrada")
    st.markdown("Selecciona un escenario base o ingresa tus propios valores.")

    escenario_seleccionado = st.selectbox(
        "Escenario de referencia",
        list(escenarios.keys()),
        index=1
    )

    valores = escenarios[escenario_seleccionado]

    st.markdown("---")

    with st.form("formulario_prediccion"):
        st.markdown("### Datos generales")
        anio = st.number_input("Año", min_value=2017, max_value=2030, value=valores["anio"], step=1)

        st.markdown("### Cobertura y operaciones")
        nro_corresponsales = st.number_input("Número de corresponsales", min_value=0, value=valores["nro_corresponsales"], step=1)
        nro_total = st.number_input("Número total de operaciones", min_value=0.0, value=valores["nro_total"], step=100.0)
        monto_total = st.number_input("Monto total de operaciones", min_value=0.0, value=valores["monto_total"], step=1000000.0)

        st.markdown("### Ahorro")
        nro_total_cta_ahorros = st.number_input("Número total de cuentas de ahorro", min_value=0.0, value=valores["nro_total_cta_ahorros"], step=50.0)
        saldo_total_cta_ahorros = st.number_input("Saldo total de cuentas de ahorro", min_value=0.0, value=valores["saldo_total_cta_ahorros"], step=1000000.0)

        st.markdown("### Crédito")
        nro_total_credito_consumo = st.number_input("Número total de créditos de consumo", min_value=0, value=valores["nro_total_credito_consumo"], step=10)
        monto_total_credito_consumo = st.number_input("Monto total de crédito de consumo", min_value=0.0, value=valores["monto_total_credito_consumo"], step=1000000.0)

        st.markdown("### Microcrédito y diversidad")
        nro_total_microcredito = st.number_input("Número total de microcréditos", min_value=0, value=valores["nro_total_microcredito"], step=5)
        monto_total_microcredito = st.number_input("Monto total de microcrédito", min_value=0.0, value=valores["monto_total_microcredito"], step=500000.0)
        cantidad_entidades = st.number_input("Cantidad de entidades financieras", min_value=0, value=valores["cantidad_entidades"], step=1)
        cantidad_tipos_producto = st.number_input("Cantidad de tipos de producto", min_value=0, value=valores["cantidad_tipos_producto"], step=1)

        boton = st.form_submit_button("Generar predicción")

    st.markdown("---")
    st.caption("Ejercicio académico. No corresponde a una medición oficial.")

# ============================================================
# CONTENIDO PRINCIPAL
# ============================================================

st.markdown('<div class="section-title">Resumen de entrada</div>', unsafe_allow_html=True)

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

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Corresponsales</div>
            <div class="metric-value">{formato_numero(nro_corresponsales)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Operaciones</div>
            <div class="metric-value">{formato_numero(nro_total)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Entidades</div>
            <div class="metric-value">{formato_numero(cantidad_entidades)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">Tipos de producto</div>
            <div class="metric-value">{formato_numero(cantidad_tipos_producto)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================================
# RESULTADOS
# ============================================================

if boton:
    nivel_predicho, tabla_probabilidades, indice = predecir_nivel(datos_usuario)
    prob_max = tabla_probabilidades["Probabilidad (%)"].max()

    st.markdown('<div class="section-title">Resultado del modelo</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1.4])

    with r1:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)

        if nivel_predicho == "alto":
            st.markdown('<div class="result-high">NIVEL ALTO</div>', unsafe_allow_html=True)
        elif nivel_predicho == "medio":
            st.markdown('<div class="result-medium">NIVEL MEDIO</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-low">NIVEL BAJO</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)

        with m1:
            st.metric("Índice calculado", round(indice, 4))

        with m2:
            st.metric("Confianza principal", f"{prob_max:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="result-panel">', unsafe_allow_html=True)
        st.markdown("#### Probabilidades por categoría")
        st.markdown(bloque_probabilidades(tabla_probabilidades), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Interpretación</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="interpretation">
            {interpretar_resultado(nivel_predicho, prob_max)}
            <br><br>
            Esta predicción debe entenderse como una aproximación académica basada en los datos ingresados 
            y en el índice construido durante el proyecto.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Detalle de probabilidades</div>', unsafe_allow_html=True)
    st.dataframe(
        tabla_probabilidades[["Nivel", "Probabilidad (%)"]],
        use_container_width=True
    )

else:
    st.markdown('<div class="section-title">Cómo usar la aplicación</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">1. Selecciona un escenario</div>
                <div class="card-text">
                    Usa un escenario base o modifica manualmente los valores desde el panel lateral.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">2. Ingresa variables</div>
                <div class="card-text">
                    Completa datos de operaciones, corresponsales, ahorro, crédito y microcrédito.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">3. Genera la predicción</div>
                <div class="card-text">
                    El modelo entrega un nivel estimado y las probabilidades para cada categoría.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# NOTA METODOLÓGICA
# ============================================================

with st.expander("Ver nota metodológica del modelo"):
    st.write(
        """
        El modelo fue construido como un ejercicio académico de Deep Learning. 
        Primero se depuró y agregó la información por municipio y fecha de corte. 
        Luego se creó un índice de inclusión financiera a partir de variables de acceso, uso y diversidad financiera. 
        Finalmente, se entrenó una red neuronal para predecir el nivel del siguiente corte.

        La clasificación bajo, medio o alto no corresponde a una categoría oficial, sino a una variable objetivo 
        construida para fines académicos a partir del dataset disponible.
        """
    )

st.markdown(
    """
    <div class="footer">
        Proyecto final de Deep Learning | Modelo de inclusión financiera municipal en Colombia
    </div>
    """,
    unsafe_allow_html=True
)
