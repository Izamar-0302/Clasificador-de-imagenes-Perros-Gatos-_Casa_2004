import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# Configuración de la página
# =====================================================

st.set_page_config(
    page_title="Clasificador de Gatos y Perros IA_ISC",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CSS — diseño limpio con líneas, sin emojis
# =====================================================

st.markdown("""
<style>
html, body, [class*="css"], * {
    font-family: 'Times New Roman', Times, serif !important;
}

.stApp {
    background: #F5F7FA;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    max-width: 700px !important;
}

/* ── ENCABEZADO ── */
.app-header {
    border-top: 4px solid #1D4ED8;
    border-bottom: 1px solid #CBD5E1;
    padding: 28px 0 20px;
    margin-bottom: 28px;
}
.app-header-tag {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1D4ED8;
    font-weight: 700;
    margin-bottom: 8px;
}
.app-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 4px 0;
    line-height: 1.2;
}
.app-header-sub {
    font-size: 0.88rem;
    color: #64748B;
    margin: 0;
}

/* ── TABLA DE PROYECTO ── */
.project-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 28px;
    font-size: 0.875rem;
}
.project-table caption {
    text-align: left;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1D4ED8;
    font-weight: 700;
    padding-bottom: 8px;
}
.project-table td {
    padding: 9px 12px;
    border-bottom: 1px solid #E2E8F0;
    color: #334155;
}
.project-table td:first-child {
    color: #94A3B8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: 38%;
    border-right: 1px solid #E2E8F0;
}
.project-table tr:last-child td {
    border-bottom: none;
}

/* ── IMAGEN ── */
.img-frame {
    border: 1px solid #CBD5E1;
    border-top: 3px solid #1D4ED8;
    margin-bottom: 24px;
    background: #fff;
}
.img-frame img { display: block; width: 100%; }

/* ── RESULTADO ── */
.result-block {
    border: 1px solid #E2E8F0;
    border-left: 4px solid #1D4ED8;
    background: #FFFFFF;
    padding: 20px 24px;
    margin-bottom: 20px;
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
    line-height: 1;
}
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.80rem;
    color: #475569;
    margin-bottom: 6px;
}
.bar-track {
    background: #E2E8F0;
    height: 6px;
    width: 100%;
}
.bar-fill {
    height: 100%;
    background: #1D4ED8;
    animation: growBar 0.8s ease-out forwards;
}
@keyframes growBar { from { width: 0%; } }

/* ── ESTADO VACÍO ── */
.empty-state {
    border: 1px dashed #CBD5E1;
    padding: 48px 24px;
    text-align: center;
    color: #94A3B8;
    font-size: 0.9rem;
    margin-top: 8px;
    background: #fff;
}

/* ── SUBIR IMAGEN LABEL ── */
.upload-tag {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #1D4ED8;
    font-weight: 700;
    margin-bottom: 6px;
    display: block;
    border-top: 1px solid #E2E8F0;
    padding-top: 20px;
}

/* ── RESPONSIVE ── */
@media (max-width: 480px) {
    .app-header-title { font-size: 1.3rem; }
    .result-name { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# ENCABEZADO
# =====================================================

st.markdown("""
<div class="app-header">
    <div class="app-header-tag">Inteligencia Artificial — ISC</div>
    <h1 class="app-header-title">Clasificador de Gatos y Perros</h1>
    <p class="app-header-sub">Modelo Predictivo con MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INFORMACION DEL PROYECTO
# =====================================================

st.markdown("""
<table class="project-table">
    <caption>Informacion del Proyecto</caption>
    <tr>
        <td>Estudiante</td>
        <td>Angeles Euceda</td>
    </tr>
    <tr>
        <td>Numero de Cuenta</td>
        <td>20221930061</td>
    </tr>
    <tr>
        <td>Clase</td>
        <td>Inteligencia Artificial — ISC</td>
    </tr>
    <tr>
        <td>Campus / Ano</td>
        <td>Comayagua, 2026</td>
    </tr>
</table>
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
# SUBIR IMAGEN
# =====================================================

st.markdown('<span class="upload-tag">Subir imagen</span>', unsafe_allow_html=True)

archivo = st.file_uploader(
    "Seleccione una imagen JPG, JPEG o PNG",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# RESULTADO
# =====================================================

if archivo is not None:
    imagen = Image.open(archivo)

    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(imagen, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown("""
<div class="empty-state">
    Sube una imagen para comenzar la clasificacion.
</div>
""", unsafe_allow_html=True)
