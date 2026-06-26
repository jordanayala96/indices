import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador de Capacidad Exportadora",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para ocultar menús por defecto de Streamlit y dar look institucional
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .st-emotion-cache-1y4p8pa {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CARGA DE DATOS Y ENTRENAMIENTO CACHEADO
# ==========================================
@st.cache_resource
def load_and_train_model():
    # Cargar los datos reales exportados del Notebook
    try:
        X_ml = pd.read_csv('X_completo.csv')
        y_ml = pd.read_csv('y_completo.csv')

        # Asegurarse de que y_ml sea un vector unidimensional
        if isinstance(y_ml, pd.DataFrame):
            y_ml = y_ml.iloc[:, 0]

        # Nos aseguramos de quitar la constante si se exportó desde statsmodels
        if 'const' in X_ml.columns:
            X_ml = X_ml.drop(columns=['const'])

    except FileNotFoundError:
        st.error(
            "🚨 Error: No se encontraron los archivos 'X_completo.csv' y 'y_completo.csv'. Asegúrate de que estén en la misma carpeta.")
        st.stop()

    # Entrenamiento del modelo XGBoost
    modelo_xgb = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    modelo_xgb.fit(X_ml, y_ml)

    # Entrenar explainer SHAP
    explainer = shap.TreeExplainer(modelo_xgb)

    return modelo_xgb, explainer, X_ml.columns, X_ml


modelo_xgb, explainer, columnas_modelo, X_referencia = load_and_train_model()

# ==========================================
# 3. INTERFAZ DE USUARIO - BARRA LATERAL (NAVEGACIÓN E INPUTS)
# ==========================================
st.sidebar.image("https://www.ecuadorencifras.gob.ec/wp-content/uploads/2026/03/banner-enesem-2024.jpg", width=200)

# Menú de Navegación principal
st.sidebar.title("Panel de selección")
pagina_actual = st.sidebar.radio(
    "Seleccione la pestaña:",
    ["Predicción de exportación", "Explicabilidad SHAP"]
)

st.sidebar.markdown("---")

# Controles interactivos
st.sidebar.title("Parámetros a configurar")
st.sidebar.markdown("Ajuste las características para realizar la simulación.")

idl_input = st.sidebar.slider("Índice de Desempeño Laboral (IDL)", min_value=-3.0, max_value=3.0, value=0.5, step=0.1)
personal_input = st.sidebar.number_input("Personal Ocupado (Total)", min_value=100, max_value=5000, value=300)
prop_calificado_input = st.sidebar.slider("Proporción Personal Calificado (%)", 0.0, 100.0, 20.0) / 100.0
pct_extranjero_input = st.sidebar.slider("Capital Extranjero (%)", 0.0, 100.0, 0.0) / 100.0
margen_input = st.sidebar.slider("Margen de Valor Agregado (%)", 0.0, 100.0, 30.0) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("Caracteristicas Demográficas")
macrosector_input = st.sidebar.selectbox("Sector Económico",
                                         ["Manufactura", "Minería", "Comercio", "Construcción", "Servicios"])
region_input = st.sidebar.selectbox("Región Geográfica", ["Sierra", "Costa", "Amazonía", "Insular"])
tic3_1_input = st.sidebar.checkbox("¿Realiza ventas directas por internet?")

# ==========================================
# 4. PROCESAMIENTO DE INPUTS AL FORMATO DEL MODELO
# ==========================================
input_data = {col: X_referencia[col].median() for col in columnas_modelo}

input_data['IDL'] = idl_input
input_data['Log_Personal'] = np.log(personal_input) if personal_input > 0 else 0
input_data['Prop_Calificado'] = prop_calificado_input
input_data['Pct_Capital_Ext'] = pct_extranjero_input
input_data['Margen_ValAg'] = margen_input
input_data['tic3_1'] = 1.0 if tic3_1_input else 0.0

sectores = ['MacroSector_1.0', 'MacroSector_2.0', 'MacroSector_3.0', 'MacroSector_4.0', 'MacroSector_5.0']
regiones = ['Region_1.0', 'Region_2.0', 'Region_3.0', 'Region_4.0']

for sec in sectores:
    if sec in input_data: input_data[sec] = 0.0
for reg in regiones:
    if reg in input_data: input_data[reg] = 0.0

mapa_sector = {"Manufactura": "MacroSector_1.0", "Minería": "MacroSector_2.0", "Comercio": "MacroSector_3.0",
               "Construcción": "MacroSector_4.0", "Servicios": "MacroSector_5.0"}
mapa_region = {"Sierra": "Region_1.0", "Costa": "Region_2.0", "Amazonía": "Region_3.0", "Insular": "Region_4.0"}

if mapa_sector[macrosector_input] in input_data:
    input_data[mapa_sector[macrosector_input]] = 1.0
if mapa_region[region_input] in input_data:
    input_data[mapa_region[region_input]] = 1.0

# Convertir a DataFrame (1 sola fila)
df_input = pd.DataFrame([input_data])[columnas_modelo]

# Realizar la predicción global para ambas vistas
probabilidad = modelo_xgb.predict_proba(df_input)[0][1] * 100

# ==========================================
# 5. LÓGICA DE VISTAS (PÁGINAS)
# ==========================================

if pagina_actual == "Predicción de exportación":
    st.title("Simulador de Autoselección Exportadora")
    st.markdown(
        "Texto Pendiente")
    st.markdown("<br>", unsafe_allow_html=True)

    # Construir el Gauge Chart (Velocímetro) con Plotly
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad,
        number={'suffix': "%", 'font': {'size': 60, 'color': '#1f3a60'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilidad de Exportar", 'font': {'size': 26, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f3a60"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#e74c3c"},  # Rojo (Baja propensión)
                {'range': [30, 70], 'color': "#f1c40f"},  # Amarillo (Media)
                {'range': [70, 100], 'color': "#2ecc71"}  # Verde (Alta)
            ]
        }
    ))

    # Aumentar la altura para que luzca imponente
    fig_gauge.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))

    # Usamos columnas para centrar el gráfico en la pantalla ya que quitamos los textos de la derecha
    col_izq, col_centro, col_der = st.columns([1, 2, 1])

    with col_centro:
        st.plotly_chart(fig_gauge, use_container_width=True)

elif pagina_actual == "Explicabilidad SHAP":
    st.title("Explicabilidad con SHAP Values")
    st.markdown(
        "Descripción Pendiente".format(
            probabilidad))

    # Generar Valores SHAP para la predicción actual
    shap_values_local = explainer(df_input)

    # Crear la figura para atrapar el plot de matplotlib
    fig_shap, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values_local[0], show=False, max_display=10)
    plt.tight_layout()

    # Mostrar en Streamlit
    st.pyplot(fig_shap)

# Footer Global
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px; margin-top: 50px; border-top: 1px solid #ddd; padding-top: 15px;">
    Encuesta Estructural Empresarial - ENESEM 2024 <br>
    Texto Pendiente
</div>
""", unsafe_allow_html=True)