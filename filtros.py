import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


# ======================================================
# UTILIDADES DE MÁSCARA / PINCEL
# ======================================================
def normalizar_mascara(mascara):
    """Convierte cualquier máscara a uint8 binaria: 0 = sano, 255 = zona a procesar."""
    if mascara is None:
        raise ValueError("La máscara no puede ser None.")

    mask = np.asarray(mascara)

    if mask.ndim == 3:
        # Si viene RGBA, normalmente el canal alpha indica trazos.
        if mask.shape[2] == 4:
            mask = mask[:, :, 3]
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return mask


def refinar_mascara_pintada(mascara, expansion=2, cierre=1, apertura=0):
    """
    Refina una máscara dibujada por el usuario.
    - cierre: une pequeños cortes en el trazo.
    - apertura: elimina puntos aislados si aparecen.
    - expansion: cubre el grosor real de la grieta alrededor del trazo.
    """
    mask = normalizar_mascara(mascara)

    if apertura > 0:
        k = 2 * apertura + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if cierre > 0:
        k = 2 * cierre + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if expansion > 0:
        k = 2 * expansion + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def mascara_tiene_contenido(mascara):
    """Devuelve True si el usuario pintó al menos algunos píxeles."""
    return mascara is not None and int(np.count_nonzero(mascara)) > 0


def crear_alpha_suave(mascara, suavizado_borde=3):
    """
    Crea un alpha flotante para mezclar filtros locales.
    No se usa como máscara de inpainting; se usa para que los filtros no dejen bordes duros.
    """
    mask = normalizar_mascara(mascara).astype(np.float32) / 255.0

    if suavizado_borde > 0:
        k = 2 * suavizado_borde + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0.0, 1.0)

    return mask[:, :, None]


def mezclar_por_mascara(img_rgb, img_procesada_rgb, mascara, suavizado_borde=3):
    """Mezcla una imagen procesada con la original usando únicamente la zona pintada."""
    alpha = crear_alpha_suave(mascara, suavizado_borde)
    salida = img_rgb.astype(np.float32) * (1.0 - alpha) + img_procesada_rgb.astype(np.float32) * alpha
    return np.clip(salida, 0, 255).astype(np.uint8)


def superponer_mascara(img_rgb, mascara, intensidad=0.45):
    """Genera una vista de depuración con la máscara en rojo sobre la imagen."""
    mask = normalizar_mascara(mascara)
    salida = img_rgb.copy().astype(np.float32)
    rojo = np.zeros_like(salida)
    rojo[:, :, 0] = 255
    alpha = (mask[:, :, None] > 0).astype(np.float32) * intensidad
    salida = salida * (1 - alpha) + rojo * alpha
    return np.clip(salida, 0, 255).astype(np.uint8)


# ======================================================
# ETAPA 1: GEOMETRÍA Y MORFOLOGÍA - GRIETAS
# ======================================================
def reparar_grietas_mascara(
    img_rgb,
    mascara_pintada,
    metodo_inpaint="Telea",
    radio_inpaint=3,
    expansion=2,
    cierre=1,
    apertura=0,
):
    """
    Repara grietas usando una máscara pintada por el usuario.
    Esta versión reemplaza la ROI rectangular: solo se reconstruyen los píxeles marcados.
    """
    mascara_refinada = refinar_mascara_pintada(
        mascara_pintada,
        expansion=expansion,
        cierre=cierre,
        apertura=apertura,
    )

    if not mascara_tiene_contenido(mascara_refinada):
        return img_rgb.copy(), mascara_refinada

    flag = cv2.INPAINT_TELEA if metodo_inpaint == "Telea" else cv2.INPAINT_NS
    restaurada = cv2.inpaint(img_rgb, mascara_refinada, inpaintRadius=radio_inpaint, flags=flag)
    return restaurada, mascara_refinada


def detectar_grietas_automatico_conservador(
    img_rgb,
    tipo="Oscuras",
    kernel_size=15,
    umbral=25,
    min_area=8,
    max_area=6000,
    proteger_detalle=True,
):
    """
    Detección automática conservadora de grietas mediante morfología.
    - tipo='Oscuras': black-hat para grietas/rayones oscuros.
    - tipo='Claras': top-hat para rayones claros.
    Se conserva como apoyo, pero el modo recomendado para rostros es el pincel.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_suave = cv2.GaussianBlur(gray, (3, 3), 0)

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if tipo == "Claras":
        realce = cv2.morphologyEx(gray_suave, cv2.MORPH_TOPHAT, kernel)
    else:
        realce = cv2.morphologyEx(gray_suave, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(realce, umbral, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica inicial.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

    # Filtro por área de componentes conectados.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtrada = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtrada[labels == i] = 255

    if proteger_detalle:
        # Excluye regiones de muy alto detalle para evitar ojos, cejas, cabello, texto fino, etc.
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        detalle = cv2.convertScaleAbs(lap)
        _, detalle_mask = cv2.threshold(detalle, 45, 255, cv2.THRESH_BINARY)
        detalle_mask = cv2.dilate(detalle_mask, k3, iterations=1)
        filtrada = cv2.bitwise_and(filtrada, cv2.bitwise_not(detalle_mask))

    return filtrada, realce


# Compatibilidad con tu versión anterior, por si quieres comparar resultados.
def reparar_grietas(img_rgb, metodo_inpaint, umbral_canny1=50, umbral_canny2=150, dilatacion=3):
    """Versión anterior: Canny + dilatación + inpainting. Útil solo como comparación."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bordes = cv2.Canny(gray, umbral_canny1, umbral_canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilatacion, dilatacion))
    mascara = cv2.dilate(bordes, kernel, iterations=1)
    flag = cv2.INPAINT_TELEA if metodo_inpaint == "Telea" else cv2.INPAINT_NS
    img_restaurada = cv2.inpaint(img_rgb, mascara, inpaintRadius=3, flags=flag)
    return img_restaurada, mascara


# ======================================================
# ETAPA 2: OPERACIONES ENTRE PÍXELES
# ======================================================
def ajustar_iluminacion_contraste(img_rgb, gamma=1.0, clip_limit=2.0):
    """Aplica corrección gamma y CLAHE en luminancia LAB."""
    gamma = max(float(gamma), 0.01)
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_gamma = cv2.LUT(img_rgb, tabla)

    lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_procesado = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_procesado, cv2.COLOR_LAB2RGB)


# ======================================================
# ETAPA 3: IDENTIFICACIÓN DE COLOR HSV
# ======================================================
def corregir_manchas_sepia(img_rgb, hue_min=10, hue_max=40, sat_reducida=0.3):
    """Detecta tonos sepia/humedad en HSV y reduce su saturación."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_8u = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    limite_inf = np.array([hue_min, 40, 40])
    limite_sup = np.array([hue_max, 255, 255])
    mascara = cv2.inRange(hsv_8u, limite_inf, limite_sup)

    hsv[:, :, 1] = np.where(mascara > 0, hsv[:, :, 1] * float(sat_reducida), hsv[:, :, 1])
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ======================================================
# ETAPA 4: FILTRADO DIGITAL
# ======================================================
def filtro_mediana(img_rgb, kernel_size=3):
    """Remueve ruido impulsivo tipo sal y pimienta preservando bordes mejor que un promedio."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img_rgb, kernel_size)


def filtro_bilateral(img_rgb, diametro=7, sigma_color=50, sigma_espacio=50):
    """Filtrado preservador de bordes. Útil como alternativa suave al promedio/Gaussiano."""
    return cv2.bilateralFilter(img_rgb, int(diametro), float(sigma_color), float(sigma_espacio))


def filtro_wiener_frecuencia(img_rgb, psf_size=5, nsr=0.01):
    """Filtro de Wiener en frecuencia para contrarrestar desenfoque moderado."""

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

    if psf_size % 2 == 0:
        psf_size += 1

    r, g, b = cv2.split(img_rgb)
    return cv2.merge((procesar_canal(r), procesar_canal(g), procesar_canal(b)))


def aplicar_filtro_local(img_rgb, mascara, nombre_filtro, parametros, expansion=0, cierre=0, suavizado_borde=3):
    """
    Aplica una operación solo sobre la zona pintada.
    A diferencia del inpainting, aquí se procesa una versión completa y luego se mezcla por máscara.
    """
    mask = refinar_mascara_pintada(mascara, expansion=expansion, cierre=cierre)

    if nombre_filtro == "Contraste (CLAHE/Gamma)":
        procesada = ajustar_iluminacion_contraste(
            img_rgb,
            gamma=parametros.get("gamma", 1.0),
            clip_limit=parametros.get("clip_limit", 2.0),
        )
    elif nombre_filtro == "Corrección Sepia (HSV)":
        procesada = corregir_manchas_sepia(
            img_rgb,
            hue_min=parametros.get("hue_min", 10),
            hue_max=parametros.get("hue_max", 40),
            sat_reducida=parametros.get("sat_reducida", 0.3),
        )
    elif nombre_filtro == "Filtro Mediana":
        procesada = filtro_mediana(img_rgb, kernel_size=parametros.get("kernel_size", 3))
    elif nombre_filtro == "Filtro Bilateral":
        procesada = filtro_bilateral(
            img_rgb,
            diametro=parametros.get("diametro", 7),
            sigma_color=parametros.get("sigma_color", 50),
            sigma_espacio=parametros.get("sigma_espacio", 50),
        )
    elif nombre_filtro == "Filtro Wiener (Desenfocado)":
        procesada = filtro_wiener_frecuencia(
            img_rgb,
            psf_size=parametros.get("psf_size", 5),
            nsr=parametros.get("nsr", 0.01),
        )
    else:
        procesada = img_rgb.copy()

    salida = mezclar_por_mascara(img_rgb, procesada, mask, suavizado_borde=suavizado_borde)
    return salida, mask