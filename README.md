# Sistema de Restauración Digital de Fotografías Degradadas

## Ejecución

```bash
pip install streamlit opencv-python numpy pillow scipy
streamlit run app.py
```

## Cambio principal implementado

La herramienta de daños físicos ahora usa un flujo híbrido clásico:

1. Detección de grietas claras con white top-hat.
2. Detección de grietas oscuras con black-hat.
3. Apoyo opcional con Canny.
4. Limpieza con apertura, cierre y dilatación morfológica.
5. Clasificación por componentes conectados:
   - grietas delgadas,
   - grietas gruesas / rasgaduras,
   - huecos / esquinas.
6. Relleno según tipo:
   - interpolación orientada para grietas delgadas,
   - inpainting clásico para rasgaduras,
   - clonación por parche para huecos y esquinas.

No se usan técnicas de machine learning, deep learning ni IA generativa.
