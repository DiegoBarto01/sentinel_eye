import numpy as np
import cv2

#Modulo 2 Estabilidad Estructural y Ajuste Dinámico
'''
Las cámaras en minería vibran y se mueven. 
● Tarea A (Vibración): Calcular y graficar en tiempo real el desplazamiento (Eje X, Eje Y) de la 
cámara respecto a una referencia estable. Identificar si hay una vibración constante (patrón de 
fallo mecánico). 
● Tarea B (Ajuste de ROI - "El desafío Senior"): 
○ Define una "Región de Interés" (ROI) inicial (ej. un recuadro sobre una zona específica del 
video). 
○ Si la cámara se mueve (pan/tilt no intencional por viento), el algoritmo debe re-calcular 
automáticamente la posición de la ROI para que siga cubriendo la misma zona física de 
la imagen. 
○ Objetivo: Que la ROI "persiga" al objetivo estático aunque la cámara se mueva. 
'''

# --- Cálculo de desplazamiento (ORB + affine transform) ---
def calculate_shift(reference_frame, current_frame):
    gray_ref = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None:
        return 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return 0, 0

    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:50]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        return 0, 0

    dx, dy = M[0, 2], M[1, 2]
    return dx, dy


# --- Ajuste Dinámico de ROI ---
def adjust_roi(roi, shift):
    x, y, w, h = roi
    dx, dy = shift

    # filtro pequeño para suavizar jitter
    dx = int(dx * 0.8)
    dy = int(dy * 0.8)

    return (x + dx, y + dy, w, h)


# --- Dibujo OSD: Gráfico de vibración ---
def draw_osd_graph(frame, dx_history, dy_history, max_points=150):
    h_osd = 120
    w_osd = 250

    osd = np.zeros((h_osd, w_osd, 3), dtype=np.uint8)

    # recorta historial
    list_dx = list(dx_history)
    list_dy = list(dy_history)
    
    dx_plot = list_dx[-max_points:]
    dy_plot = list_dy[-max_points:]

    cx = len(dx_plot)

    # normalización para que los movimientos grandes no exploten el gráfico
    scale = 2  # ajustar para agrandar/reducir amplitud

    # centro vertical
    mid_y = h_osd // 2

    # dibujar ejes
    cv2.line(osd, (0, mid_y), (w_osd, mid_y), (80, 80, 80), 1)

    # dibujar dx (verde)
    for i in range(1, cx):
        cv2.line(
            osd,
            (i-1, int(mid_y - dx_plot[i-1] * scale)),
            (i,   int(mid_y - dx_plot[i] * scale)),
            (0, 255, 0),
            1
        )

    # dibujar dy (rojo)
    for i in range(1, cx):
        cv2.line(
            osd,
            (i-1, int(mid_y - dy_plot[i-1] * scale)),
            (i,   int(mid_y - dy_plot[i] * scale)),
            (0, 0, 255),
            1
        )

    # incrustar OSD en esquina inferior derecha
    fh, fw = frame.shape[:2]
    frame[fh - h_osd - 10: fh - 10, fw - w_osd - 10: fw - 10] = osd

    return frame

# --- Analizador del video completo ---
def process_frame_stability(reference_frame, frame, roi, original_roi, dx_history, dy_history):
    # Calcular shift ORB
    dx, dy = calculate_shift(reference_frame, frame)

    # Guardar historial
    dx_history.append(dx)
    dy_history.append(dy)

    # Actualizar ROI dinámico
    new_roi = adjust_roi(roi, (dx, dy))
    x, y, w, h = new_roi

    # == DIBUJOS EN EL FRAME ==
    # ROI original (azul)
    ox, oy, ow, oh = original_roi
    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)

    # ROI ajustada (verde)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Gráfico OSD en esquina inferior derecha
    frame = draw_osd_graph(frame, dx_history, dy_history)

    return frame, new_roi, (dx, dy)

def analyze_video_stability(video_path, initial_roi):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video: {video_path}")

    ret, reference_frame = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame.")

    roi = initial_roi
    original_roi = initial_roi

    dx_history = []
    dy_history = []
    shifts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processed, roi, shift = process_frame_stability(
            reference_frame,
            frame,
            roi,
            original_roi,
            dx_history,
            dy_history
        )

        shifts.append(shift)

        cv2.imshow("Estabilidad Estructural + ROI Dinámica + OSD", frame_processed)
        if cv2.waitKey(1) & 0xFF == 27:  # tecla ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    return shifts

#video_path = r"F:\Diego\Descargas\delirio\video_2.mp4"
#video_path = r"F:\Diego\Descargas\delirio\video2_corto.mp4"
##video_path = r"F:\Diego\Descargas\delirio\24 (2025-05-12 00'00'34 - 2025-05-12 00'03'07).avi"
##video_path = r"F:\Diego\Descargas\delirio\24 (2025-05-12 08'22'13 - 2025-05-12 08'22'53).avi"
##video_path = r"F:\Diego\Descargas\delirio\video_1.mp4"
#
#cap = cv2.VideoCapture(video_path)
#ret, frame = cap.read()
#if not ret:
#    raise RuntimeError("No se pudo leer el primer frame del video.")
#
## dimensiones
#height, width = frame.shape[:2]
#
## tamaño del ROI (ajustable)
#roi_w, roi_h = 50, 50
#
## centrado en el frame
#initial_roi = (
#    width//2 - roi_w//2,
#    height//2 - roi_h//2,
#    roi_w,
#    roi_h
#)
#
#print("ROI inicial centrado:", initial_roi)
#shifts = analyze_video_stability(video_path, initial_roi)