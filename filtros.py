import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2
from skimage.filters import frangi


# ==========================================
# ETAPA 1: Geometría y Morfología (Reparación)
# ==========================================

def rellenar_huecos_esquinas(img_rgb, metodo_inpaint, umbral=200, invert_threshold=False, morph_size_close=7,
                             morph_size_open=3, inpaint_radius=5):
    """
    Detecta y rellena huecos grandes (como esquinas rotas) mediante operaciones morfológicas.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Umbralización para aislar el vacío
    thresh_type = cv2.THRESH_BINARY_INV if invert_threshold else cv2.THRESH_BINARY
    _, mascara_huecos = cv2.threshold(gray, umbral, 255, thresh_type)

    # 2. Cierre (Unir partes desconectadas)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size_close, morph_size_close))
    mascara_unida = cv2.morphologyEx(mascara_huecos, cv2.MORPH_CLOSE, kernel_close)

    # 3. Apertura (Limpiar ruido)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size_open, morph_size_open))
    mascara_limpia = cv2.morphologyEx(mascara_unida, cv2.MORPH_OPEN, kernel_open)

    # 4. Filtrado por área para evitar falsos positivos
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_final = np.zeros_like(mascara_limpia)

    area_minima = (img_rgb.shape[0] * img_rgb.shape[1]) * 0.005  # 0.5% del área total

    for cnt in contornos:
        if cv2.contourArea(cnt) > area_minima:
            cv2.drawContours(mascara_final, [cnt], -1, 255, -1)

    # 5. Inpainting
    flag = cv2.INPAINT_TELEA if metodo_inpaint == 'Telea' else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara_final, inpaintRadius=inpaint_radius, flags=flag)

    return img_restaurada, mascara_final


def reparar_grietas_avanzado(img_rgb, metodo_inpaint, metodo_deteccion='Frangi', grosor_kernel=7, umbral=30,
                             dilatar_final=3):
    """
    Detecta grietas usando algoritmos discriminatorios de forma (Frangi/Morph) para evitar rostros.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mascara = np.zeros_like(gray)

    if metodo_deteccion == 'Black-Hat':
        # Ideal para extraer líneas oscuras sobre fondos claros
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grosor_kernel, grosor_kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mascara = cv2.threshold(blackhat, umbral, 255, cv2.THRESH_BINARY)

    elif metodo_deteccion == 'Frangi':
        # Análisis de matriz Hessiana para detectar estructuras tubulares/lineales
        frangi_img = frangi(gray, black_ridges=True)
        # Normalizar al rango 0-255 uint8
        frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mascara = cv2.threshold(frangi_norm, umbral, 255, cv2.THRESH_BINARY)

    elif metodo_deteccion == 'Canny':
        # Método clásico global
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        mascara = cv2.Canny(blurred, umbral, umbral * 3)  # Usamos umbral como base para Canny

    # Dilatación para abarcar el grosor total de la grieta física
    if dilatar_final > 1:
        kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatar_final, dilatar_final))
        mascara_final = cv2.dilate(mascara, kernel_dilatacion, iterations=1)
    else:
        mascara_final = mascara

    # Inpainting
    flag = cv2.INPAINT_TELEA if metodo_inpaint == 'Telea' else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara_final, inpaintRadius=3, flags=flag)

    return img_restaurada, mascara_final


# ==========================================
# ETAPA 2: Operaciones entre píxeles
# ==========================================

def ajustar_iluminacion_contraste(img_rgb, gamma=1.0, clip_limit=2.0):
    """Aplica Corrección Gamma y Ecualización Adaptativa (CLAHE) en el canal de Luminancia."""
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
    """Detecta tonos sepia mediante rangos HSV y reduce su saturación."""
    hsv_8u = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    limite_inf = np.array([hue_min, 40, 40])
    limite_sup = np.array([hue_max, 255, 255])
    mascara = cv2.inRange(hsv_8u, limite_inf, limite_sup)

    hsv_float = hsv_8u.astype(np.float32)
    hsv_float[:, :, 1] = np.where(mascara > 0, hsv_float[:, :, 1] * sat_reducida, hsv_float[:, :, 1])
    hsv_final = np.clip(hsv_float, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2RGB)


# ==========================================
# ETAPA 4: Filtrado Digital (Espacial y Frecuencia)
# ==========================================

def filtro_mediana(img_rgb, kernel_size=3):
    """Remueve ruido sal y pimienta."""
    if kernel_size % 2 == 0: kernel_size += 1
    return cv2.medianBlur(img_rgb, kernel_size)


def filtro_wiener_frecuencia(img_rgb, psf_size=5, nsr=0.01):
    """Filtro de Wiener en el dominio de la frecuencia contra el desenfoque."""

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