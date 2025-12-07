# Sentinel Eye – Video Analytics Pipeline

Monitoreo automatizado de calidad de cámara, estabilidad y movimiento

Sentinel Eye es un pipeline de análisis de video diseñado para evaluar la calidad visual de cámaras instaladas en entornos reales. El sistema detecta suciedad, vibraciones y movimiento mediante un conjunto de módulos optimizados para funcionar en tiempo real.

Este proyecto fue desarrollado con foco en robustez, velocidad, modularidad y capacidad de despliegue usando Docker.

# Características principales

Quality Control (QC): detección de suciedad, empañamiento y baja calidad mediante histogramas y contraste.

Detección de Movimiento: basada en background subtraction + morfología para máxima velocidad.

Detección de Vibración / Estabilidad: seguimiento de un ROI dinámico que permite medir microdesplazamientos.

Pipeline modular: cada analizador funciona como módulo independiente.

Optimizado para CPU: permite correr en hardware limitado.

Ejecutable en Docker: portable y reproducible.
