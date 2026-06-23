import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


# ==========================================
# ETAPA 1: Geometría y Morfología (Grietas)
# ==========================================
def reparar_grietas(img_rgb, metodo_inpaint, umbral_canny1=50, umbral_canny2=150, dilatacion=3, mascara_proteccion=None):
    """
    Detecta grietas mediante Canny y transformaciones morfológicas.
    Soporta una máscara de protección para evitar deformar rostros.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gray, umbral_canny1, umbral_canny2)

    # Si hay protección, se apagan los bordes detectados en el rostro
    if mascara_proteccion is not None:
        bordes = cv2.bitwise_and(bordes, bordes, mask=mascara_proteccion)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatacion, dilatacion))
    mascara = cv2.dilate(bordes, kernel, iterations=1)

    flag = cv2.INPAINT_TELEA if metodo_inpaint == 'Telea' else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara, inpaintRadius=3, flags=flag)

    return img_restaurada, mascara


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
    # Normalizar la máscara de 0-255 a rango [0.0, 1.0]
    alpha = mascara_proteccion.astype(np.float32) / 255.0

    if difuminar:
        if kernel_suavizado % 2 == 0: kernel_suavizado += 1
        # Aplicar desenfoque al canal alpha para transiciones suaves (Modo 3)
        alpha = cv2.GaussianBlur(alpha, (kernel_suavizado, kernel_suavizado), 0)

    # Expandir de 2D a 3D para multiplicar por los canales RGB
    alpha = np.expand_dims(alpha, axis=-1)

    # Ecuación de mezcla matricial:
    # Donde alpha es 1 (Modificable) queda procesada. Donde alpha es 0 (Rostro) queda original.
    img_final = (img_procesada * alpha) + (img_original * (1.0 - alpha))
    return np.clip(img_final, 0, 255).astype(np.uint8)