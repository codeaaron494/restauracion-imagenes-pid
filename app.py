import streamlit as st
import cv2
import numpy as np
from PIL import Image
import filtros

st.set_page_config(layout="wide")
st.title("Sistema de Restauración Digital de Fotografías Degradadas")

# 1. Carga de archivo
uploaded_file = st.file_uploader("Sube una fotografía histórica degradada (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 2. Lógica de session_state para mantener la imagen a través de las iteraciones
    if 'img_original' not in st.session_state:
        # Convertir archivo a matriz de numpy
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.img_original = np.array(image)
        st.session_state.img_actual = np.copy(st.session_state.img_original)

    img_actual = st.session_state.img_actual
    h, w, _ = img_actual.shape

    st.sidebar.header("Panel de Control PID")

    # 3. Selección de Alcance (ROI manual con sliders)
    st.sidebar.subheader("Región de Interés (ROI)")
    aplicar_toda = st.sidebar.checkbox("Aplicar a toda la imagen", value=True)

    if aplicar_toda:
        y1, y2, x1, x2 = 0, h, 0, w
    else:
        y1, y2 = st.sidebar.slider("Eje Y (Alto)", 0, h, (0, h))
        x1, x2 = st.sidebar.slider("Eje X (Ancho)", 0, w, (0, w))

    # Extraer el sub-fragmento
    roi = img_actual[y1:y2, x1:x2]

    # 4. Selectbox con las operaciones del pipeline
    operacion = st.sidebar.selectbox(
        "Herramienta de Restauración",
        ["Ninguna", "Reparación de Grietas", "Contraste (CLAHE/Gamma)",
         "Corrección Sepia (HSV)", "Filtro Mediana", "Filtro Wiener (Desenfocado)"]
    )

    # Copia de seguridad para no alterar directamente el estado hasta dar clic en "Aplicar"
    roi_procesada = np.copy(roi)
    mascara_debug = None

    # 5. Ejecución del filtro seleccionado usando filtros.py
    if operacion == "Reparación de Grietas":
        metodo = st.sidebar.radio("Método Inpainting", ["Telea", "Navier-Stokes"])
        u1 = st.sidebar.slider("Umbral Canny Min", 10, 200, 50)
        u2 = st.sidebar.slider("Umbral Canny Max", 50, 300, 150)
        dil = st.sidebar.slider("Grosor de Grieta (Kernel)", 1, 10, 3)
        roi_procesada, mascara_debug = filtros.reparar_grietas(roi, metodo, u1, u2, dil)

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

    # 6. Reintegración de la ROI a la imagen clonada (Vista Previa)
    img_preview = np.copy(img_actual)
    img_preview[y1:y2, x1:x2] = roi_procesada

    # Dibujar un rectángulo rojo suave en la vista previa para identificar la ROI
    if not aplicar_toda:
        cv2.rectangle(img_preview, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Visualización
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estado Actual")
        st.image(img_actual, use_column_width=True)
    with col2:
        st.subheader("Vista Previa (Filtro Aplicado)")
        st.image(img_preview, use_column_width=True)

    if mascara_debug is not None:
        st.subheader("Modo Depuración: Máscara de Grietas Generada (Canny + Morfología)")
        st.image(mascara_debug, width=400, caption="Blanco: Daño detectado, Negro: Píxeles sanos")

    # 7. Botones de Control de Estado
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button(" Aplicar cambio permanentemente"):
            # Para guardar sin las líneas rojas del rectángulo
            img_final = np.copy(img_actual)
            img_final[y1:y2, x1:x2] = roi_procesada
            st.session_state.img_actual = img_final
            st.rerun()

    with col_btn2:
        if st.button(" Reiniciar desde original"):
            st.session_state.img_actual = np.copy(st.session_state.img_original)
            st.rerun()

    with col_btn3:
        # Descarga de la imagen final
        img_bgr = cv2.cvtColor(st.session_state.img_actual, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        st.download_button(label=" Descargar Imagen Restaurada",
                           data=buffer.tobytes(),
                           file_name="restauracion_final.jpg",
                           mime="image/jpeg")