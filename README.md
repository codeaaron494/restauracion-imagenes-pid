# Sistema de Restauración Digital con Pincel

Esta versión reemplaza la ROI rectangular por una máscara pintada por el usuario.
La idea principal es que los filtros se apliquen solamente sobre la zona dañada marcada, evitando suavizar ojos, cejas, brillos u otros detalles finos.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Flujo recomendado

1. Sube la imagen.
2. Selecciona la herramienta.
3. Pinta encima de la grieta, rasguño, mancha o zona afectada.
4. Ajusta el tamaño del pincel y la expansión morfológica.
5. Revisa la máscara y la vista previa.
6. Aplica el cambio solo cuando el resultado sea correcto.

Para rostros, usa expansión baja y evita pintar zonas sanas como cejas, pestañas o brillos de los ojos.
