import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "Falta instalar streamlit-drawable-canvas. Ejecuta: "
        "pip install streamlit-drawable-canvas"
    )
    st.stop()

import filtros


st.set_page_config(layout="wide")
st.title("Sistema de Restauración Digital de Fotografías Degradadas")
st.caption("Versión con máscara pintada: ya no se usa ROI rectangular para reparar zonas delicadas.")


# ======================================================
# UTILIDADES DE INTERFAZ
# ======================================================
def inicializar_estado():
    valores = {
        "img_original": None,
        "img_actual": None,
        "uploaded_id": None,
        "canvas_key": 0,
        "historial": [],
    }
    for clave, valor in valores.items():
        if clave not in st.session_state:
            st.session_state[clave] = valor


def calcular_vista(img_rgb, ancho_max=900):
    h, w = img_rgb.shape[:2]
    escala = min(ancho_max / w, 1.0)
    vista_w = int(w * escala)
    vista_h = int(h * escala)
    img_vista = cv2.resize(img_rgb, (vista_w, vista_h), interpolation=cv2.INTER_AREA)
    return img_vista, vista_w, vista_h


def extraer_mascara_pincel(canvas_result, fondo_vista_rgb, forma_original):
    """
    Obtiene la máscara pintada comparando el canvas con la imagen de fondo.
    Esta técnica evita depender del recorte rectangular y recupera solo el trazo del usuario.
    """
    h_orig, w_orig = forma_original[:2]

    if canvas_result.image_data is None:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)

    canvas_rgba = canvas_result.image_data.astype(np.uint8)
    canvas_rgb = canvas_rgba[:, :, :3]

    # Diferencia entre la imagen de fondo y el canvas con pintura roja.
    diff = cv2.absdiff(canvas_rgb, fondo_vista_rgb)
    mask_vista = (np.max(diff, axis=2) > 12).astype(np.uint8) * 255

    # Limpieza leve por antialiasing del pincel.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_vista = cv2.morphologyEx(mask_vista, cv2.MORPH_CLOSE, kernel)

    mask_original = cv2.resize(mask_vista, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return mask_original


def codificar_jpg(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen final.")
    return buffer.tobytes()


inicializar_estado()

uploaded_file = st.file_uploader(
    "Sube una fotografía histórica degradada (JPG, PNG o JPEG)",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is None:
    st.info("Sube una imagen para comenzar la restauración.")
    st.stop()

uploaded_id = f"{uploaded_file.name}_{uploaded_file.size}"
if st.session_state.uploaded_id != uploaded_id:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.session_state.img_original = img_np
    st.session_state.img_actual = img_np.copy()
    st.session_state.uploaded_id = uploaded_id
    st.session_state.historial = []
    st.session_state.canvas_key += 1

img_actual = st.session_state.img_actual
img_vista, vista_w, vista_h = calcular_vista(img_actual, ancho_max=900)
fondo_pil = Image.fromarray(img_vista)


# ======================================================
# PANEL DE CONTROL
# ======================================================
st.sidebar.header("Panel de Control PID")
st.sidebar.subheader("Modo de selección")
st.sidebar.markdown(
    "Pinta directamente sobre la grieta, mancha o zona dañada. "
    "El filtro se aplicará solo en esa máscara, no en un rectángulo completo."
)

operacion = st.sidebar.selectbox(
    "Herramienta de restauración",
    [
        "Reparación de Grietas (Inpainting)",
        "Contraste (CLAHE/Gamma)",
        "Corrección Sepia (HSV)",
        "Filtro Mediana",
        "Filtro Bilateral",
        "Filtro Wiener (Desenfocado)",
        "Detección automática conservadora de grietas",
    ],
)

ancho_pincel = st.sidebar.slider("Tamaño del pincel", 1, 40, 8)
expansion = st.sidebar.slider("Expansión morfológica de máscara", 0, 8, 2)
cierre = st.sidebar.slider("Cierre morfológico de máscara", 0, 5, 1)
suavizado_borde = st.sidebar.slider("Suavizado del borde del filtro", 0, 10, 3)

parametros = {}
mascara_debug = None
realce_debug = None

if operacion == "Reparación de Grietas (Inpainting)":
    metodo = st.sidebar.radio("Método de inpainting", ["Telea", "Navier-Stokes"])
    radio_inpaint = st.sidebar.slider("Radio de inpainting", 1, 10, 3)

elif operacion == "Contraste (CLAHE/Gamma)":
    parametros["gamma"] = st.sidebar.slider("Corrección gamma", 0.5, 3.0, 1.0, 0.1)
    parametros["clip_limit"] = st.sidebar.slider("Límite CLAHE", 1.0, 5.0, 2.0, 0.1)

elif operacion == "Corrección Sepia (HSV)":
    parametros["hue_min"] = st.sidebar.slider("Hue mínimo", 0, 179, 10)
    parametros["hue_max"] = st.sidebar.slider("Hue máximo", 0, 179, 40)
    parametros["sat_reducida"] = st.sidebar.slider("Factor de saturación", 0.0, 1.0, 0.3)

elif operacion == "Filtro Mediana":
    parametros["kernel_size"] = st.sidebar.slider("Tamaño kernel impar", 3, 15, 3, step=2)

elif operacion == "Filtro Bilateral":
    parametros["diametro"] = st.sidebar.slider("Diámetro", 3, 15, 7, step=2)
    parametros["sigma_color"] = st.sidebar.slider("Sigma color", 10, 150, 50)
    parametros["sigma_espacio"] = st.sidebar.slider("Sigma espacio", 10, 150, 50)

elif operacion == "Filtro Wiener (Desenfocado)":
    parametros["psf_size"] = st.sidebar.slider("Tamaño PSF", 3, 15, 5, step=2)
    parametros["nsr"] = st.sidebar.slider("NSR", 0.001, 0.100, 0.010, 0.001)

elif operacion == "Detección automática conservadora de grietas":
    tipo_auto = st.sidebar.radio("Tipo de grieta/rayón", ["Oscuras", "Claras"])
    kernel_auto = st.sidebar.slider("Tamaño de elemento estructurante", 5, 41, 15, step=2)
    umbral_auto = st.sidebar.slider("Umbral de realce", 5, 100, 25)
    proteger_detalle = st.sidebar.checkbox("Proteger regiones de alto detalle", value=True)
    metodo_auto = st.sidebar.radio("Método inpainting automático", ["Telea", "Navier-Stokes"], key="metodo_auto")
    radio_auto = st.sidebar.slider("Radio inpainting automático", 1, 8, 3, key="radio_auto")


# ======================================================
# CANVAS DE PINTADO
# ======================================================
col_canvas, col_info = st.columns([2, 1])

with col_canvas:
    st.subheader("1. Pinta la zona dañada")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=ancho_pincel,
        stroke_color="rgba(255, 0, 0, 0.80)",
        background_image=fondo_pil,
        update_streamlit=True,
        height=vista_h,
        width=vista_w,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

with col_info:
    st.subheader("Uso recomendado")
    st.write(
        "Para grietas sobre rostros, pinta solo la línea dañada. "
        "Evita cubrir brillos de ojos, pestañas, cejas o bordes reales."
    )
    st.write(
        "La expansión morfológica debe ser baja: normalmente 1 a 3 es suficiente. "
        "Si la máscara invade detalles sanos, reduce expansión o tamaño del pincel."
    )

mascara_pintada = extraer_mascara_pincel(canvas_result, img_vista, img_actual.shape)


# ======================================================
# PROCESAMIENTO / VISTA PREVIA
# ======================================================
img_preview = img_actual.copy()

if operacion == "Detección automática conservadora de grietas":
    mascara_auto, realce_debug = filtros.detectar_grietas_automatico_conservador(
        img_actual,
        tipo=tipo_auto,
        kernel_size=kernel_auto,
        umbral=umbral_auto,
        proteger_detalle=proteger_detalle,
    )
    img_preview, mascara_debug = filtros.reparar_grietas_mascara(
        img_actual,
        mascara_auto,
        metodo_inpaint=metodo_auto,
        radio_inpaint=radio_auto,
        expansion=expansion,
        cierre=cierre,
    )
else:
    if filtros.mascara_tiene_contenido(mascara_pintada):
        if operacion == "Reparación de Grietas (Inpainting)":
            img_preview, mascara_debug = filtros.reparar_grietas_mascara(
                img_actual,
                mascara_pintada,
                metodo_inpaint=metodo,
                radio_inpaint=radio_inpaint,
                expansion=expansion,
                cierre=cierre,
            )
        else:
            img_preview, mascara_debug = filtros.aplicar_filtro_local(
                img_actual,
                mascara_pintada,
                operacion,
                parametros,
                expansion=expansion,
                cierre=cierre,
                suavizado_borde=suavizado_borde,
            )
    else:
        st.warning("Aún no has pintado ninguna zona. Dibuja sobre el área dañada para generar la vista previa.")
        mascara_debug = np.zeros(img_actual.shape[:2], dtype=np.uint8)


# ======================================================
# VISUALIZACIÓN DE ETAPAS
# ======================================================
st.subheader("2. Vista previa y depuración")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Estado actual**")
    st.image(img_actual, use_container_width=True)

with col2:
    st.markdown("**Máscara aplicada**")
    st.image(filtros.superponer_mascara(img_actual, mascara_debug), use_container_width=True)

with col3:
    st.markdown("**Vista previa procesada**")
    st.image(img_preview, use_container_width=True)

if realce_debug is not None:
    st.markdown("**Mapa de realce morfológico automático**")
    st.image(realce_debug, clamp=True, use_container_width=True)


# ======================================================
# BOTONES DE CONTROL
# ======================================================
st.subheader("3. Confirmar edición")
btn1, btn2, btn3, btn4 = st.columns(4)

with btn1:
    aplicar = st.button("✅ Aplicar cambio")

with btn2:
    limpiar = st.button("🧽 Limpiar pincel")

with btn3:
    deshacer = st.button("↩️ Deshacer")

with btn4:
    reiniciar = st.button("🔄 Reiniciar original")

if aplicar:
    st.session_state.historial.append(st.session_state.img_actual.copy())
    st.session_state.historial = st.session_state.historial[-10:]
    st.session_state.img_actual = img_preview.copy()
    st.session_state.canvas_key += 1
    st.rerun()

if limpiar:
    st.session_state.canvas_key += 1
    st.rerun()

if deshacer:
    if st.session_state.historial:
        st.session_state.img_actual = st.session_state.historial.pop()
        st.session_state.canvas_key += 1
        st.rerun()
    else:
        st.info("No hay cambios anteriores para deshacer.")

if reiniciar:
    st.session_state.img_actual = st.session_state.img_original.copy()
    st.session_state.historial = []
    st.session_state.canvas_key += 1
    st.rerun()

st.download_button(
    label="📥 Descargar imagen restaurada",
    data=codificar_jpg(st.session_state.img_actual),
    file_name="restauracion_final.jpg",
    mime="image/jpeg",
)
