import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# Configuracion de la pagina
# =====================================================

st.set_page_config(
    page_title="Clasificador de Gatos y Perros IA_ISC",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CSS — todo en un solo bloque, sin HTML mezclado
# =====================================================

st.markdown("""
<style>
html, body, [class*="css"], * {
    font-family: 'Times New Roman', Times, serif !important;
}
.stApp {
    background: #F5F7FA;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    max-width: 700px !important;
}

/* Linea azul arriba de toda la pagina */
.stApp::before {
    content: '';
    display: block;
    height: 4px;
    background: #1D4ED8;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}

/* Titulos de Streamlit */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    border-bottom: 1px solid #CBD5E1;
    padding-bottom: 12px;
    margin-bottom: 4px !important;
}

/* Barra de confianza */
.bar-track {
    background: #E2E8F0;
    height: 6px;
    width: 100%;
    margin-top: 6px;
}
.bar-fill {
    height: 100%;
    background: #1D4ED8;
    animation: growBar 0.8s ease-out forwards;
}
@keyframes growBar { from { width: 0%; } }

.result-block {
    border-left: 4px solid #1D4ED8;
    background: #FFFFFF;
    padding: 20px 24px;
    margin-top: 16px;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #1D4ED8;
}
.result-tag {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 6px;
}
.result-name {
    font-size: 2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 20px;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Configuracion
# =====================================================

IMG_SIZE = (224, 224)
MODEL_DIR = Path("modelo_gatos_perros")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [
    MODEL_DIR / "gatos_perros_mobilenet.keras",
    MODEL_DIR / "gatos_perros_mobilenet.h5"
]
LABELS_ES = {"Gatos": "Gato", "Perros": "Perro"}

# =====================================================
# Cargar modelo y clases
# =====================================================

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontro el modelo. Coloque la carpeta 'modelo_gatos_perros' junto al archivo app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Gatos", "Perros"]

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    predicciones = modelo.predict(preparar_imagen(img), verbose=0)[0]
    indice = np.argmax(predicciones)
    clase = clases[indice]
    confianza = float(predicciones[indice]) * 100
    return clase, confianza

modelo = cargar_modelo()
clases = cargar_clases()

# =====================================================
# INTERFAZ — solo componentes nativos de Streamlit
# =====================================================

st.title("Clasificador de Gatos y Perros")
st.caption("Modelo Predictivo con MobileNetV2  |  Inteligencia Artificial ISC")

st.divider()

col1, col2 = st.columns(2)
col1.metric("Estudiante", "Angeles Euceda")
col1.metric("Numero de Cuenta", "20221930061")
col2.metric("Clase", "IA - ISC")
col2.metric("Campus / Ano", "Comayagua 2026")

st.divider()

archivo = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, use_container_width=True)

    clase, confianza = predecir(imagen)
    nombre = LABELS_ES.get(clase, clase)

    st.markdown(f"""
<div class="result-block">
    <div class="result-tag">Resultado de la Clasificacion</div>
    <div class="result-name">{nombre}</div>
    <div class="conf-row">
        <span>Confianza del modelo</span>
        <span><strong>{confianza:.1f}%</strong></span>
    </div>
    <div class="bar-track">
        <div class="bar-fill" style="width:{confianza:.1f}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.info("Sube una imagen para comenzar la clasificacion.")
