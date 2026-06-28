import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Gatos y Perros IA_ISC",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
# INTERFAZ — 100% componentes nativos, cero HTML
# =====================================================

st.title("Clasificador de Gatos y Perros")
st.caption("Modelo Predictivo con MobileNetV2  |  Inteligencia Artificial ISC")

st.divider()

col1, col2 = st.columns(2)
col1.metric("Estudiante", "Angeles Euceda")
col1.metric("Numero de Cuenta", "20221930061")
col2.metric("Clase", "IA - ISC")
col2.metric("Campus / Año", "Comayagua 2026")

st.divider()

archivo = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, use_container_width=True)

    clase, confianza = predecir(imagen)
    nombre = LABELS_ES.get(clase, clase)

    st.divider()
    st.subheader("Resultado")
    st.write(f"**Prediccion:** {nombre}")
    st.write("**Confianza del modelo**")
    st.progress(confianza / 100, text=f"{confianza:.1f}%")

else:
    st.info("Sube una imagen para comenzar la clasificacion.")
