import cv2
import time

VIDEO_PATH = r"./video.mp4" #Ajustar a tu path para probar independientemente

#Modulo 3  Detección de Movimiento Optimizada
'''
Eficiencia computacional es sostenibilidad. 
● Tarea: Implementar un detector de movimiento o de objetos (puedes usar solamente 
Background Subtraction avanzado) para detectar actividad en la zona. 
● Requisito Senior: Esta inferencia NO debe matar el rendimiento. Debes demostrar 
optimización. 
○ Uso de Batch Processing (si aplica). 
○ Uso de TensorRT/ONNX para acelerar la inferencia. 
○ O implementación de lógica de salto de frames (procesar N frames por segundo) sin 
perder precisión crítica. 
● Output: Bounding boxes sobre los objetos/movimiento detectados.
'''

# --- CONFIGURACIÓN GLOBAL ---
FRAME_SKIP = 1           
THRESHOLD_AREA = 800     

# --- Inicialización de Objetos Permanentes (MOG2 y CLAHE) ---
FGBG_MODEL = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=True)
CLAHE_MODEL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# --------------------------------------------------------------------------------

def detectar_movimiento_frame(frame, fgbg_model, clahe_model, threshold_area):
    """
    Calcula y detecta el movimiento en un solo frame usando Sustracción de Fondo MOG2 
    con pre-procesamiento de robustez a iluminación (Normalización y CLAHE).

    :param frame: El frame de entrada (RGB/BGR).
    :param fgbg_model: El objeto cv2.BackgroundSubtractorMOG2 inicializado.
    :param clahe_model: El objeto cv2.CLAHE inicializado.
    :param threshold_area: El área mínima del contorno para ser considerada movimiento.
    :return: El frame original con Bounding Boxes dibujadas.
    """
    # Copia del frame original para el output
    original_frame = frame.copy() 

    # --- Pre-procesamiento para Detección Mejorada (Niebla/Oscuridad/Robustez a Luz) ---

    # 1. Conversión a escala de grises y Normalización de Intensidad
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalización para robustez contra cambios de iluminación global
    normalized_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Aplicación de CLAHE (Contraste Adaptativo)
    cl1 = clahe_model.apply(normalized_frame)

    # 3. Reducción de Ruido Gaussiano (Suavizado)
    blurred_frame = cv2.GaussianBlur(cl1, (5, 5), 0)

    processed_frame = blurred_frame 
    
    # 4. Aplicar Sustracción de Fondo al frame PRE-PROCESADO
    fgmask = fgbg_model.apply(processed_frame)

    # 5. Procesamiento de la Máscara de Primer Plano (Operaciones Morfológicas)
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=4)

    # 6. Encontrar Contornos (Movimiento)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Dibujar Bounding Boxes sobre el frame ORIGINAL
    for contour in contours:
        if cv2.contourArea(contour) < threshold_area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Dibujar la Bounding Box en el frame ORIGINAL
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Color ROJO

    return original_frame

# --------------------------------------------------------------------------------

#Función para probar módulo independientemente
def correr_deteccion_video(video_path, fgbg_model, clahe_model, frame_skip=1, threshold_area=800):
    """
    Corre el código principal de detección de movimiento, gestionando la captura de video, 
    la lógica de salto de frames y la visualización.

    :param video_path: La ruta al archivo de video.
    :param fgbg_model: El objeto cv2.BackgroundSubtractorMOG2 inicializado.
    :param clahe_model: El objeto cv2.CLAHE inicializado.
    :param frame_skip: Número de frames a saltar por cada frame procesado (optimización).
    :param threshold_area: Área mínima de contorno para la detección.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return

    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            frame_count += 1

            # Lógica de Salto de Frames (Optimización de Rendimiento)
            if frame_count % frame_skip != 0:
                continue

            # Llamada a la función de detección de movimiento
            frame_con_deteccion = detectar_movimiento_frame(
                frame, 
                fgbg_model, 
                clahe_model, 
                threshold_area
            )

            # Mostrar Resultados
            cv2.imshow('Deteccion de Movimiento Optimizada', frame_con_deteccion)
            
            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- Finalización y Métricas ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        processed_frames = frame_count // frame_skip

        print("---")
        print("Proceso finalizado.")
        print(f"Frames leídos del video: {frame_count}")
        print(f"Frames procesados por la lógica de detección: {processed_frames}")
        print(f"Tiempo total de procesamiento: {elapsed_time:.2f} segundos")
        if elapsed_time > 0:
            print(f"Tasa de Frames Procesados (FPS): {processed_frames / elapsed_time:.2f} FPS")

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

# --------------------------------------------------------------------------------
## Ejecución del Código
#correr_deteccion_video(
#        video_path=VIDEO_PATH, 
#        fgbg_model=FGBG_MODEL, 
#        clahe_model=CLAHE_MODEL, 
#        frame_skip=1, # Puedes cambiar el valor de salto de frames aquí
#        threshold_area=800  # Puedes cambiar el área mínima aquí

#    )
