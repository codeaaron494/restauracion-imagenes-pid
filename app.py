import streamlit as st
import cv2
import numpy as np
from PIL import Image
import filtros

st.set_page_config(layout="wide", page_title="Restauración PID Clásico")
st.title("Sistema de Restauración Digital de Fotografías Degradadas")
st.markdown("Procesamiento Digital de Imágenes estricto mediante métodos matemáticos, sin IA.")

# 1. Carga de archivo
uploaded_file = st.file_uploader("Sube una fotografía histórica degradada (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 2. Lógica de session_state
    if 'img_original' not in st.session_state:
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.img_original = np.array(image)
        st.session_state.img_actual = np.copy(st.session_state.img_original)

    img_actual = st.session_state.img_actual
    h, w, _ = img_actual.shape

    st.sidebar.header("Panel de Control PID")

    # 3. Región de Interés (ROI)
    st.sidebar.subheader("Selección de Alcance (ROI)")
    aplicar_toda = st.sidebar.checkbox("Aplicar a toda la imagen", value=True)

    if aplicar_toda:
        y1, y2, x1, x2 = 0, h, 0, w
    else:
        y1, y2 = st.sidebar.slider("Eje Y (Alto)", 0, h, (0, h))
        x1, x2 = st.sidebar.slider("Eje X (Ancho)", 0, w, (0, w))
        if y1 == y2: y2 += 1
        if x1 == x2: x2 += 1

    roi = img_actual[y1:y2, x1:x2]

    # 4. Selectbox con todas las operaciones
    operacion = st.sidebar.selectbox(
        "Herramienta Matemática",
        ["Ninguna",
         "Relleno de Huecos (Esquinas)",
         "Reparación de Grietas Avanzada",
         "Contraste (CLAHE/Gamma)",
         "Corrección Sepia (HSV)",
         "Filtro Mediana",
         "Filtro Wiener (Desenfocado)"]
    )

    roi_procesada = np.copy(roi)
    mascara_debug = None

    # 5. Ejecución con controles avanzados
    if operacion == "Relleno de Huecos (Esquinas)":
        metodo_h = st.sidebar.radio("Método Inpainting", ["Telea", "Navier-Stokes"])
        u_hueco = st.sidebar.slider("Umbral de Detección (Intensidad)", 0, 255, 200)
        invert_u = st.sidebar.checkbox("Invertir Umbral (¿Hueco oscuro?)", value=False)
        k_close = st.sidebar.slider("Kernel Cierre (Unir partes)", 3, 31, 7, step=2)
        k_open = st.sidebar.slider("Kernel Apertura (Limpiar ruido)", 1, 15, 3, step=2)
        r_inpaint = st.sidebar.slider("Radio Inpainting", 1, 15, 5)

        roi_procesada, mascara_debug = filtros.rellenar_huecos_esquinas(
            roi, metodo_h, u_hueco, invert_u, k_close, k_open, r_inpaint
        )

    elif operacion == "Reparación de Grietas Avanzada":
        metodo_g = st.sidebar.radio("Método Inpainting", ["Telea", "Navier-Stokes"], key="inpaint_grietas")
        metodo_det = st.sidebar.selectbox("Algoritmo de Detección", ["Frangi", "Black-Hat", "Canny"])

        umbral_g = st.sidebar.slider("Sensibilidad (Umbral)", 1, 150, 30)
        k_grosor = st.sidebar.slider("Grosor Referencia Kernel (Black-Hat)", 3, 21, 7, step=2)
        dil = st.sidebar.slider("Dilatación Final (Grosor Relleno)", 1, 10, 3)

        roi_procesada, mascara_debug = filtros.reparar_grietas_avanzado(
            roi, metodo_g, metodo_det, k_grosor, umbral_g, dil
        )

    elif operacion == "Contraste (CLAHE/Gamma)":
        gamma = st.sidebar.slider("Corrección Gamma", 0.5, 3.0, 1.0, 0.1)
        clip = st.sidebar.slider("Límite CLAHE", 1.0, 5.0, 2.0, 0.1)
        roi_procesada = filtros.ajustar_iluminacion_contraste(roi, gamma, clip)

    elif operacion == "Corrección Sepia (HSV)":
        h_min = st.sidebar.slider("Matiz (Hue) Min", 0, 179, 10)
        h_max = st.sidebar.slider("Matiz (Hue) Max", 0, 179, 40)
        sat = st.sidebar.slider("Factor de Saturación", 0.0, 1.0, 0.3)
        roi_procesada = filtros.corregir_manchas_sepia(roi, h_min, h_max, sat)

    elif operacion == "Filtro Mediana":
        k_size = st.sidebar.slider("Tamaño Kernel (Impar)", 3, 15, 3, step=2)
        roi_procesada = filtros.filtro_mediana(roi, k_size)

    elif operacion == "Filtro Wiener (Desenfocado)":
        psf = st.sidebar.slider("Tamaño de Desenfoque (PSF)", 3, 15, 5, step=2)
        nsr = st.sidebar.slider("Relación Ruido-Señal (NSR)", 0.001, 0.1, 0.01, 0.001)
        roi_procesada = filtros.filtro_wiener_frecuencia(roi, psf, nsr)

    # 6. Reintegración y Visualización
    img_preview = np.copy(img_actual)
    img_preview[y1:y2, x1:x2] = roi_procesada

    if not aplicar_toda:
        cv2.rectangle(img_preview, (x1, y1), (x2, y2), (255, 0, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estado Actual")
        st.image(img_actual, use_container_width=True)
    with col2:
        st.subheader("Vista Previa (Filtro Aplicado)")
        st.image(img_preview, use_container_width=True)

    if mascara_debug is not None:
        st.subheader("Modo Depuración: Máscara Binaria Aislada")
        st.image(mascara_debug, width=400, caption="Blanco: Región detectada para restaurar")

    # 7. Botones de Control de Estado
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button(" Aplicar cambio permanentemente", use_container_width=True):
            img_final = np.copy(img_actual)
            img_final[y1:y2, x1:x2] = roi_procesada
            st.session_state.img_actual = img_final
            st.rerun()

    with col_btn2:
        if st.button(" Reiniciar desde original", use_container_width=True):
            st.session_state.img_actual = np.copy(st.session_state.img_original)
            st.rerun()

    with col_btn3:
        img_bgr = cv2.cvtColor(st.session_state.img_actual, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        st.download_button(label="📥 Descargar Imagen Final",
                           data=buffer.tobytes(),
                           file_name="restauracion_PID.jpg",
                           mime="image/jpeg",
                           use_container_width=True)