import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


# ==========================================
# ETAPA 1: Geometría y Morfología (Grietas)
# ==========================================
def reparar_grietas(img_rgb, metodo_inpaint, umbral_canny1=50, umbral_canny2=150, dilatacion=3):
    """
    Detecta grietas mediante Canny y transformaciones morfológicas,
    luego reconstruye la zona con Telea o Navier-Stokes.
    """
    # 1. Escala de grises para aislar el gradiente de intensidad
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Detección de bordes (Aislamiento de la grieta)
    bordes = cv2.Canny(gray, umbral_canny1, umbral_canny2)

    # 3. Dilatación morfológica para abarcar el grosor real del rayón
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatacion, dilatacion))
    mascara = cv2.dilate(bordes, kernel, iterations=1)

    # 4. Aplicación de Inpainting clásico
    flag = cv2.INPAINT_TELEA if metodo_inpaint == 'Telea' else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara, inpaintRadius=3, flags=flag)

    return img_restaurada, mascara


# ==========================================
# ETAPA 2: Operaciones entre píxeles
# ==========================================
def ajustar_iluminacion_contraste(img_rgb, gamma=1.0, clip_limit=2.0):
    """Aplica Corrección Gamma y Ecualización Adaptativa (CLAHE) en el canal de Luminancia."""
    # Corrección Gamma (curva de mapeo no lineal)
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_rgb, tabla)

    # CLAHE (Se aplica en espacio LAB, canal L, para no corromper los colores)
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

    # Rango típico para humedad/oxidación del papel
    limite_inf = np.array([hue_min, 40, 40])
    limite_sup = np.array([hue_max, 255, 255])

    # Segmentación
    mascara = cv2.inRange(hsv_8u, limite_inf, limite_sup)

    # Desaturación selectiva mediante álgebra matricial
    hsv[:, :, 1] = np.where(mascara > 0, hsv[:, :, 1] * sat_reducida, hsv[:, :, 1])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ==========================================
# ETAPA 4: Filtrado Digital (Espacial y Frecuencia)
# ==========================================
def filtro_mediana(img_rgb, kernel_size=3):
    """Remueve ruido sal y pimienta preservando bordes."""
    # OpenCV requiere que el kernel sea un número impar
    if kernel_size % 2 == 0: kernel_size += 1
    return cv2.medianBlur(img_rgb, kernel_size)


def filtro_wiener_frecuencia(img_rgb, psf_size=5, nsr=0.01):
    """
    Filtro de Wiener en el dominio de la frecuencia mediante Transformada de Fourier.
    Contrarresta el desenfoque (Point Spread Function) equilibrando la señal-ruido.
    """

    def procesar_canal(canal):
        canal_float = canal.astype(np.float64)

        # 1. Definir la Función de Dispersión de Punto (PSF) de desenfoque
        psf = np.ones((psf_size, psf_size)) / (psf_size * psf_size)

        # 2. Rellenar la PSF al tamaño de la imagen y centrarla
        psf_padded = np.zeros_like(canal_float)
        h, w = psf.shape
        psf_padded[:h, :w] = psf
        psf_padded = np.roll(psf_padded, -h // 2, axis=0)
        psf_padded = np.roll(psf_padded, -w // 2, axis=1)

        # 3. Transformada Rápida de Fourier (FFT 2D)
        G = fft2(canal_float)
        H = fft2(psf_padded)

        # 4. Ecuación Matemática del Filtro de Wiener
        H_conj = np.conj(H)
        H_mag_sq = np.abs(H) ** 2
        # F(u,v) = [H*(u,v) / (|H(u,v)|^2 + NSR)] * G(u,v)
        W = H_conj / (H_mag_sq + nsr)

        F_hat = G * W

        # 5. Transformada Inversa para volver al dominio espacial
        f_hat = np.real(ifft2(F_hat))
        return np.clip(f_hat, 0, 255).astype(np.uint8)

    # Se aplica canal por canal al ser RGB
    r, g, b = cv2.split(img_rgb)
    return cv2.merge((procesar_canal(r), procesar_canal(g), procesar_canal(b)))