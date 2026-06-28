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
    layout="centered"
)

st.title("🐱🐶 Modelo Predictivo de Clasificación de Gatos y Perros")
st.write("**Clase:** Inteligencia Artificial - Ingeniería en Sistemas Computacionales")
st.write("**Campus:** Comayagua - 2026")
st.write("**Estudiante:** Angeles Euceda")
st.write("**Número de Cuenta:** 20221930061")

st.divider()

st.write(
    "Suba una imagen para clasificar si corresponde a un **Gato** o un **Perro** "
    "utilizando un modelo **MobileNetV2 preentrenado**."
)

# =====================================================
# Configuración
# =====================================================

IMG_SIZE = (224,224)

MODEL_DIR = Path("modelo_gatos_perros")

CLASS_PATH = MODEL_DIR / "class_names.json"

MODEL_PATHS = [
    MODEL_DIR / "gatos_perros_mobilenet.keras",
    MODEL_DIR / "gatos_perros_mobilenet.h5"
]

# =====================================================
# Etiquetas en español
# =====================================================

LABELS_ES = {
    "Gatos": "Gato",
    "Perros": "Perro"
}

# =====================================================
# Cargar modelo
# =====================================================

@st.cache_resource
def cargar_modelo():

    for path in MODEL_PATHS:

        if path.exists():

            return tf.keras.models.load_model(path, compile=False)

    st.error("No se encontró el modelo. Coloque la carpeta 'modelo_gatos_perros' junto al archivo app.py.")
    st.stop()

# =====================================================
# Cargar clases
# =====================================================

@st.cache_data
def cargar_clases():

    if CLASS_PATH.exists():

        with open(CLASS_PATH, "r", encoding="utf-8") as f:

            return json.load(f)

    return ["Gatos", "Perros"]

# =====================================================
# Preparar imagen
# =====================================================

def preparar_imagen(img):

    img = img.convert("RGB")

    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    return arr

# =====================================================
# Predicción
# =====================================================

def predecir(img):

    predicciones = modelo.predict(preparar_imagen(img), verbose=0)[0]

    indice = np.argmax(predicciones)

    clase = clases[indice]

    confianza = float(predicciones[indice]) * 100

    return clase, confianza

# =====================================================
# Cargar recursos
# =====================================================

modelo = cargar_modelo()

clases = cargar_clases()

# =====================================================
# Subir imagen
# =====================================================

archivo = st.file_uploader(
    "Seleccione una imagen",
    type=["jpg","jpeg","png"]
)

# =====================================================
# Realizar predicción
# =====================================================

if archivo is not None:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen seleccionada",
        use_container_width=True
    )

    clase, confianza = predecir(imagen)

    nombre = LABELS_ES.get(clase, clase)

    st.subheader("Resultado de la clasificación")

    st.success(f"Predicción: {nombre}")

    st.metric(
        label="Confianza del modelo",
        value=f"{confianza:.2f}%"
    )

else:

    st.info("Seleccione una imagen para comenzar la clasificación.")
