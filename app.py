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
    if 'img_original' not in st.session_state:
        image = Image.open(uploaded_file).convert('RGB')
        st.session_state.img_original = np.array(image)
        st.session_state.img_actual = np.copy(st.session_state.img_original)

    img_actual = st.session_state.img_actual
    h, w, _ = img_actual.shape

    st.sidebar.header("Panel de Control PID")

    # 2. Selección de Alcance (ROI)
    st.sidebar.subheader("Región de Interés (ROI)")
    aplicar_toda = st.sidebar.checkbox("Aplicar a toda la imagen", value=True)

    if aplicar_toda:
        y1, y2, x1, x2 = 0, h, 0, w
    else:
        y1, y2 = st.sidebar.slider("Eje Y (Alto)", 0, h, (0, h))
        x1, x2 = st.sidebar.slider("Eje X (Ancho)", 0, w, (0, w))

    roi = img_actual[y1:y2, x1:x2]

    # 3. Selección de Herramienta Base
    operacion = st.sidebar.selectbox(
        "Herramienta de Restauración",
        ["Ninguna", "Reparación de Grietas", "Tratamiento de Bordes Faltantes", "Contraste (CLAHE/Gamma)",
         "Corrección Sepia (HSV)", "Filtro Mediana", "Filtro Wiener (Desenfocado)"]
    )

    # Variables para parámetros de cada herramienta
    metodo_inp, u1, u2, dil = 'Telea', 50, 150, 3
    gamma, clip = 1.0, 2.0
    h_min, h_max, sat = 10, 40, 0.3
    k_size = 3
    psf, nsr = 5, 0.01

    if operacion == "Reparación de Grietas":
        metodo_inp = st.sidebar.radio("Método Inpainting", ["Telea", "Navier-Stokes"])
        u1 = st.sidebar.slider("Umbral Canny Min", 10, 200, 50)
        u2 = st.sidebar.slider("Umbral Canny Max", 50, 300, 150)
        dil = st.sidebar.slider("Grosor de Grieta (Kernel)", 1, 10, 3)


    elif operacion == "Tratamiento de Bordes Faltantes":
        modo_borde = st.sidebar.radio("Modo de Tratamiento",
                                      ["Relleno Sólido (Fondo)", "Relleno Fluido (Inpainting)",
                                       "Recorte Inteligente (Auto-Crop)"])
        color_fondo = st.sidebar.selectbox("Color de fondo del escáner", ["Blanco", "Negro"])
        if color_fondo == "Blanco":
            umbral_borde = st.sidebar.slider("Umbral de Detección", 200, 255, 240)
        else:
            umbral_borde = st.sidebar.slider("Umbral de Detección", 0, 50, 15)
        dil_borde = st.sidebar.slider("Expansión/Suavizado", 1, 15, 5)
        # Variable para el ruido solo si estamos en el modo sólido
        nivel_ruido = 0
        if modo_borde == "Relleno Sólido (Fondo)":
            nivel_ruido = st.sidebar.slider("Grano Sintético (Ruido)", 0, 50, 15,
                                            help="Simula la textura de la foto para que el parche no se vea plano")

    elif operacion == "Contraste (CLAHE/Gamma)":
        gamma = st.sidebar.slider("Corrección Gamma", 0.5, 3.0, 1.0, 0.1)
        clip = st.sidebar.slider("Límite CLAHE", 1.0, 5.0, 2.0, 0.1)

    elif operacion == "Corrección Sepia (HSV)":
        h_min = st.sidebar.slider("Matiz (Hue) Min", 0, 179, 10)
        h_max = st.sidebar.slider("Matiz (Hue) Max", 0, 179, 40)
        sat = st.sidebar.slider("Factor de Saturación", 0.0, 1.0, 0.3)

    elif operacion == "Filtro Mediana":
        k_size = st.sidebar.slider("Tamaño Kernel (Impar)", 3, 15, 3, step=2)

    elif operacion == "Filtro Wiener (Desenfocado)":
        psf = st.sidebar.slider("Tamaño de Desenfoque (PSF)", 3, 15, 5, step=2)
        nsr = st.sidebar.slider("Relación Ruido-Señal (NSR)", 0.001, 0.1, 0.01, 0.001)

    # 4. CAPA UNIVERSAL DE PROTECCIÓN (Aplica a todas las herramientas)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Protección de Rostros (Global)")
    modo_proteccion = st.sidebar.selectbox(
        "Estrategia de Protección",
        ["Ninguna", "Modo 1: Máscara Manual", "Modo 2: Detección Haar", "Modo 3: Fusión Avanzada"]
    )

    mascara_proteccion = None
    rostros_detectados = []
    px1, px2, py1, py2 = 0, 0, 0, 0
    k_suavizado = 15

    if modo_proteccion != "Ninguna":
        mascara_proteccion = np.ones(roi.shape[:2], dtype=np.uint8) * 255

        if modo_proteccion == "Modo 1: Máscara Manual":
            alto_roi, ancho_roi = roi.shape[:2]
            py1, py2 = st.sidebar.slider("Proteger Eje Y", 0, alto_roi, (int(alto_roi * 0.2), int(alto_roi * 0.8)))
            px1, px2 = st.sidebar.slider("Proteger Eje X", 0, ancho_roi, (int(ancho_roi * 0.2), int(ancho_roi * 0.8)))
            mascara_proteccion[py1:py2, px1:px2] = 0

        elif modo_proteccion in ["Modo 2: Detección Haar", "Modo 3: Fusión Avanzada"]:
            rostros_detectados = filtros.detectar_rostros_haar(roi)
            for (x, y, w, h) in rostros_detectados:
                y_ini, y_fin = max(0, y - 10), min(roi.shape[0], y + h + 10)
                x_ini, x_fin = max(0, x - 10), min(roi.shape[1], x + w + 10)
                mascara_proteccion[y_ini:y_fin, x_ini:x_fin] = 0

            if modo_proteccion == "Modo 3: Fusión Avanzada" and len(rostros_detectados) > 0:
                k_suavizado = st.sidebar.slider("Difuminado de Borde (Gaussian)", 1, 51, 15, step=2)

    # 5. MOTOR DE EJECUCIÓN
    roi_procesada = np.copy(roi)
    mascara_debug = None

    if operacion != "Ninguna":
        # PASO A: Ejecutar el algoritmo seleccionado
        if operacion == "Reparación de Grietas":
            roi_procesada, mascara_debug = filtros.reparar_grietas(roi, metodo_inp, u1, u2, dil, mascara_proteccion)
        elif operacion == "Tratamiento de Bordes Faltantes":
            roi_procesada, mascara_debug = filtros.tratar_bordes_faltantes(roi, modo_borde, color_fondo, umbral_borde,
                                                                           dil_borde, nivel_ruido)
        elif operacion == "Contraste (CLAHE/Gamma)":
            roi_procesada = filtros.ajustar_iluminacion_contraste(roi, gamma, clip)
        elif operacion == "Corrección Sepia (HSV)":
            roi_procesada = filtros.corregir_manchas_sepia(roi, h_min, h_max, sat)
        elif operacion == "Filtro Mediana":
            roi_procesada = filtros.filtro_mediana(roi, k_size)
        elif operacion == "Filtro Wiener (Desenfocado)":
            roi_procesada = filtros.filtro_wiener_frecuencia(roi, psf, nsr)

        # PASO B: Aplicar post-procesamiento de protección si está activo y no hubo recorte
        if mascara_proteccion is not None and roi_procesada.shape == roi.shape:
            difuminar_activo = (modo_proteccion == "Modo 3: Fusión Avanzada")
            roi_procesada = filtros.fusionar_capas_protegidas(roi, roi_procesada, mascara_proteccion,
                                                              difuminar=difuminar_activo, kernel_suavizado=k_suavizado)

    # 6. Reintegración y Dibujo de Guías Visuales
    # Si la imagen cambió de tamaño (por el recorte inteligente), solo mostramos la procesada
    if roi_procesada.shape != roi.shape:
        img_preview = roi_procesada
    else:
        img_preview = np.copy(img_actual)
        img_preview[y1:y2, x1:x2] = roi_procesada

        if operacion != "Ninguna":
            if modo_proteccion == "Modo 1: Máscara Manual":
                cv2.rectangle(img_preview, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
            elif modo_proteccion in ["Modo 2: Detección Haar", "Modo 3: Fusión Avanzada"]:
                for (rx, ry, rw, rh) in rostros_detectados:
                    cv2.rectangle(img_preview, (x1 + rx, y1 + ry), (x1 + rx + rw, y1 + ry + rh), (0, 255, 0), 2)

        if not aplicar_toda:
            cv2.rectangle(img_preview, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 7. Interfaz de Visualización
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estado Actual")
        st.image(img_actual, use_column_width=True)
    with col2:
        st.subheader("Vista Previa (Filtro + Protección)")
        st.image(img_preview, use_column_width=True)

    if mascara_debug is not None:
        st.subheader("Depuración: Máscara Generada")
        st.image(mascara_debug, width=400)

    # 8. Controles de Estado
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("✅ Aplicar permanentemente"):
            # Si el tamaño cambió debido al auto-crop, la imagen procesada reemplaza a la actual
            if roi_procesada.shape != roi.shape:
                st.session_state.img_actual = roi_procesada
            else:
                img_final = np.copy(img_actual)
                img_final[y1:y2, x1:x2] = roi_procesada
                st.session_state.img_actual = img_final
            st.rerun()

    with col_btn2:
        if st.button("🔄 Reiniciar desde original"):
            st.session_state.img_actual = np.copy(st.session_state.img_original)
            st.rerun()

    with col_btn3:
        img_bgr = cv2.cvtColor(st.session_state.img_actual, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        st.download_button(label="📥 Descargar Imagen Restaurada",
                           data=buffer.tobytes(),
                           file_name="restauracion_final.jpg",
                           mime="image/jpeg")