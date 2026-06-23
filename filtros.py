import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


# ==========================================
# ETAPA 1: Geometría y Morfología (Grietas)
# ==========================================
def reparar_grietas(img_rgb, metodo_inpaint, umbral_canny1=50, umbral_canny2=150, dilatacion=3,
                    mascara_proteccion=None):
    """
    Detecta grietas mediante Canny y transformaciones morfológicas.
    Soporta una máscara de protección para evitar deformar rostros.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gray, umbral_canny1, umbral_canny2)

    if mascara_proteccion is not None:
        bordes = cv2.bitwise_and(bordes, bordes, mask=mascara_proteccion)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatacion, dilatacion))
    mascara = cv2.dilate(bordes, kernel, iterations=1)

    flag = cv2.INPAINT_TELEA if metodo_inpaint == 'Telea' else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara, inpaintRadius=3, flags=flag)

    return img_restaurada, mascara


# ==========================================
# ETAPA 1.5: Tratamiento de Bordes Faltantes
# ==========================================
def tratar_bordes_faltantes(img_rgb, modo="Relleno Sólido (Fondo)", color_fondo="Blanco", umbral=240, dilatacion=5,
                            nivel_ruido=15):
    """
    Detecta zonas faltantes (huecos) mediante umbralización asumiendo un color de fondo del escáner.
    Puede rellenar de forma fluida, sólida con textura, o hacer un recorte inteligente.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Crear máscara del fondo faltante
    if color_fondo == "Blanco":
        _, mascara_fondo = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
    else:
        _, mascara_fondo = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY_INV)

    # 2. Ejecutar la acción seleccionada
    if modo == "Relleno Fluido (Inpainting)":
        # Bueno para huecos pequeños (efecto "agua" en huecos masivos)
        kernel = np.ones((dilatacion, dilatacion), np.uint8)
        mascara_dilatada = cv2.dilate(mascara_fondo, kernel, iterations=1)
        img_restaurada = cv2.inpaint(img_rgb, mascara_dilatada, 3, cv2.INPAINT_TELEA)
        return img_restaurada, mascara_dilatada

    elif modo == "Relleno Sólido (Fondo)":
        # Solución para huecos masivos: Relleno de color + Grano sintético
        img_restaurada = np.copy(img_rgb)

        # A. Crear un anillo alrededor del hueco para muestrear el color sano
        kernel_anillo = np.ones((15, 15), np.uint8)
        anillo = cv2.dilate(mascara_fondo, kernel_anillo, iterations=1) - mascara_fondo

        # B. Calcular el color promedio de ese anillo
        color_promedio = cv2.mean(img_rgb, mask=anillo)[:3]

        # C. Crear una matriz del mismo tamaño llena de ese color sólido
        capa_solida = np.full_like(img_rgb, color_promedio, dtype=np.uint8)

        # D. Generar ruido monocromático (Solo en escala de grises)
        ruido_mono = np.zeros(img_rgb.shape[:2], dtype=np.int16)
        cv2.randn(ruido_mono, 0, nivel_ruido)

        # E. Aplicar el mismo ruido a los 3 canales (R, G, B) para que sea acromático
        ruido_3d = np.stack([ruido_mono] * 3, axis=-1)
        capa_texturizada = cv2.add(capa_solida, ruido_3d, dtype=cv2.CV_8UC3)

        # F. Reemplazar SOLO los píxeles del hueco con la nueva textura
        idx_hueco = mascara_fondo > 0
        img_restaurada[idx_hueco] = capa_texturizada[idx_hueco]

        # G. Difuminar ligeramente la "costura" (el borde de la máscara)
        kernel_costura = np.ones((dilatacion, dilatacion), np.uint8)
        borde_costura = cv2.dilate(mascara_fondo, kernel_costura, iterations=1) - cv2.erode(mascara_fondo,
                                                                                            kernel_costura,
                                                                                            iterations=1)
        img_suavizada = cv2.GaussianBlur(img_restaurada, (7, 7), 0)

        # Mezclar el borde suavizado
        idx_costura = borde_costura > 0
        img_restaurada[idx_costura] = img_suavizada[idx_costura]

        return img_restaurada, mascara_fondo

    elif modo == "Recorte Inteligente (Auto-Crop)":
        mascara_foto = cv2.bitwise_not(mascara_fondo)
        contornos, _ = cv2.findContours(mascara_foto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            c_max = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c_max)
            img_recortada = img_rgb[y:y + h, x:x + w]

            mascara_debug = np.zeros_like(img_rgb)
            cv2.rectangle(mascara_debug, (x, y), (x + w, y + h), (0, 255, 0), 3)
            return img_recortada, mascara_debug

        return img_rgb, mascara_fondo

# ==========================================
# ETAPA 2: Operaciones entre píxeles
# ==========================================
def ajustar_iluminacion_contraste(img_rgb, gamma=1.0, clip_limit=2.0):
    """Aplica Corrección Gamma y Ecualización Adaptativa (CLAHE)."""
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_rgb, tabla)

    lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_procesado = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_procesado, cv2.COLOR_LAB2RGB)


# ==========================================
# ETAPA 3: Identificación de Color (HSV)
# ==========================================
def corregir_manchas_sepia(img_rgb, hue_min=10, hue_max=40, sat_reducida=0.3):
    """Detecta tonos sepia/humedad mediante rangos HSV y reduce su saturación."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_8u = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    limite_inf = np.array([hue_min, 40, 40])
    limite_sup = np.array([hue_max, 255, 255])
    mascara = cv2.inRange(hsv_8u, limite_inf, limite_sup)

    hsv[:, :, 1] = np.where(mascara > 0, hsv[:, :, 1] * sat_reducida, hsv[:, :, 1])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ==========================================
# ETAPA 4: Filtrado Digital (Espacial y Frecuencia)
# ==========================================
def filtro_mediana(img_rgb, kernel_size=3):
    """Remueve ruido sal y pimienta preservando bordes."""
    if kernel_size % 2 == 0: kernel_size += 1
    return cv2.medianBlur(img_rgb, kernel_size)


def filtro_wiener_frecuencia(img_rgb, psf_size=5, nsr=0.01):
    """Filtro de Wiener en el dominio de la frecuencia mediante FFT."""

    def procesar_canal(canal):
        canal_float = canal.astype(np.float64)
        psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)

        psf_padded = np.zeros_like(canal_float)
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        psf_padded = np.roll(psf_padded, -h // 2, axis=0)
        psf_padded = np.roll(psf_padded, -w // 2, axis=1)

        G = fft2(canal_float)
        H = fft2(psf_padded)
        H_conj = np.conj(H)
        H_mag_sq = np.abs(H) ** 2
        W = H_conj / (H_mag_sq + nsr)

        F_hat = G * W
        f_hat = np.real(ifft2(F_hat))
        return np.clip(f_hat, 0, 255).astype(np.uint8)

    r, g, b = cv2.split(img_rgb)
    return cv2.merge((procesar_canal(r), procesar_canal(g), procesar_canal(b)))


# ==========================================
# FUNCIONES DE PROTECCIÓN UNIVERSAL
# ==========================================
def detectar_rostros_haar(img_rgb):
    """Detecta rostros usando el clasificador preentrenado de OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return rostros


def fusionar_capas_protegidas(img_original, img_procesada, mascara_proteccion, difuminar=False, kernel_suavizado=15):
    """
    Motor universal de mezcla (Alpha Blending).
    Pondera la imagen original y la procesada basándose en la máscara generada en la UI.
    """
    alpha = mascara_proteccion.astype(np.float32) / 255.0

    if difuminar:
        if kernel_suavizado % 2 == 0: kernel_suavizado += 1
        alpha = cv2.GaussianBlur(alpha, (kernel_suavizado, kernel_suavizado), 0)

    alpha = np.expand_dims(alpha, axis=-1)
    img_final = (img_procesada * alpha) + (img_original * (1.0 - alpha))
    return np.clip(img_final, 0, 255).astype(np.uint8)