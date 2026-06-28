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
# CSS personalizado — diseño moderno azul/blanco
# =====================================================

st.markdown("""
<style>
/* ── Fuentes ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Fondo general ── */
.stApp {
    background: linear-gradient(160deg, #EEF4FF 0%, #F8FAFF 60%, #E8F0FE 100%);
    min-height: 100vh;
}

/* ── Ocultar elementos por defecto de Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 720px !important; }

/* ── ENCABEZADO con degradado ── */
.hero-header {
    background: linear-gradient(135deg, #1A56DB 0%, #1E40AF 40%, #1D4ED8 70%, #2563EB 100%);
    border-radius: 0 0 32px 32px;
    padding: 48px 32px 40px;
    text-align: center;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.30);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -30px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-icons {
    font-size: 56px;
    line-height: 1;
    margin-bottom: 12px;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.25));
}
.hero-title {
    color: #FFFFFF;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* ── TARJETA DE PROYECTO ── */
.project-card {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.08);
}
.project-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #2563EB;
    margin-bottom: 12px;
}
.project-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
}
.project-item {
    display: flex;
    flex-direction: column;
}
.project-label {
    font-size: 0.70rem;
    color: #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.project-value {
    font-size: 0.88rem;
    color: #111827;
    font-weight: 500;
}

/* ── SECCIÓN UPLOAD ── */
.upload-label {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #2563EB;
    margin-bottom: 8px;
    display: block;
}
.upload-area {
    background: #FFFFFF;
    border: 2px dashed #93C5FD;
    border-radius: 16px;
    padding: 8px 16px 16px;
    transition: border-color 0.2s;
    box-shadow: 0 2px 8px rgba(37,99,235,0.06);
    margin-bottom: 24px;
}
.upload-hint {
    text-align: center;
    color: #6B7280;
    font-size: 0.82rem;
    margin-top: 4px;
}
/* Streamlit file uploader overrides */
[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ── IMAGEN con borde y sombra ── */
.img-frame {
    border: 3px solid #BFDBFE;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 28px rgba(37,99,235,0.18);
    margin-bottom: 24px;
}
.img-frame img {
    display: block;
    width: 100%;
}

/* ── TARJETA RESULTADO ── */
.result-card {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(37,99,235,0.10);
    margin-bottom: 20px;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6B7280;
    margin-bottom: 8px;
}
.result-animal {
    font-size: 2rem;
    font-weight: 800;
    color: #1D4ED8;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.confidence-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}
.bar-track {
    background: #EFF6FF;
    border-radius: 99px;
    height: 12px;
    overflow: hidden;
    border: 1px solid #BFDBFE;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563EB 0%, #60A5FA 100%);
    animation: growBar 0.9s cubic-bezier(0.34,1.2,0.64,1) forwards;
    transform-origin: left;
}
@keyframes growBar {
    from { width: 0%; }
}

/* ── ESTADO VACÍO ── */
.empty-state {
    text-align: center;
    padding: 40px 24px;
    color: #9CA3AF;
}
.empty-icon { font-size: 48px; margin-bottom: 10px; }
.empty-text { font-size: 0.95rem; }

/* ── RESPONSIVE ── */
@media (max-width: 480px) {
    .hero-title { font-size: 1.35rem; }
    .hero-icons { font-size: 42px; }
    .project-grid { grid-template-columns: 1fr; }
    .hero-header { padding: 36px 20px 30px; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# ENCABEZADO con degradado
# =====================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-icons">🐱🐶</div>
    <h1 class="hero-title">Clasificador de Gatos y Perros</h1>
    <p class="hero-subtitle">Modelo Predictivo con MobileNetV2 | Inteligencia Artificial ISC</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# TARJETA DE INFORMACIÓN DEL PROYECTO
# =====================================================

st.markdown("""
<div class="project-card">
    <div class="project-card-title">📋 Información del Proyecto</div>
    <div class="project-grid">
        <div class="project-item">
            <span class="project-label">Estudiante</span>
            <span class="project-value">Angeles Euceda</span>
        </div>
        <div class="project-item">
            <span class="project-label">Número de Cuenta</span>
            <span class="project-value">20221930061</span>
        </div>
        <div class="project-item">
            <span class="project-label">Clase</span>
            <span class="project-value">Inteligencia Artificial | ISC</span>
        </div>
        <div class="project-item">
            <span class="project-label">Campus / Año</span>
            <span class="project-value">Comayagua | 2026</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Configuración
# =====================================================

IMG_SIZE = (224, 224)
MODEL_DIR = Path("modelo_gatos_perros")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [
    MODEL_DIR / "gatos_perros_mobilenet.keras",
    MODEL_DIR / "gatos_perros_mobilenet.h5"
]
LABELS_ES = {"Gatos": "Gato", "Perros": "Perro"}
ANIMAL_ICON = {"Gato": "🐱", "Perro": "🐶"}

# =====================================================
# Cargar modelo y clases
# =====================================================

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("No se encontró el modelo. Coloque la carpeta 'modelo_gatos_perros' junto al archivo app.py.")
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
# ÁREA DE CARGA
# =====================================================

st.markdown('<span class="upload-label">📤 Subir imagen</span>', unsafe_allow_html=True)
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

archivo = st.file_uploader(
    "Arrastra una imagen aquí o haz clic para seleccionarla",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">Formatos aceptados: JPG | JPEG | PNG</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RESULTADO
# =====================================================

if archivo is not None:
    imagen = Image.open(archivo)

    # Imagen con borde y sombra
    st.markdown('<div class="img-frame">', unsafe_allow_html=True)
    st.image(imagen, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    clase, confianza = predecir(imagen)
    nombre = LABELS_ES.get(clase, clase)
    icono = ANIMAL_ICON.get(nombre, "🐾")

    # Tarjeta resultado con barra animada
    st.markdown(f"""
<div class="result-card">
    <div class="result-label">Resultado de la Clasificación</div>
    <div class="result-animal">{icono} {nombre}</div>
    <div class="confidence-label">
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
    <div class="empty-icon">🖼️</div>
    <div class="empty-text">Sube una imagen para comenzar la clasificación</div>
</div>
""", unsafe_allow_html=True)
