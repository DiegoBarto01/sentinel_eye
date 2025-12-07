import cv2
import os
import argparse
import time
import numpy as np
from qc_module import process_qc_frame
from stability_module import process_frame_stability
from motion_sustract import detectar_movimiento_frame

# --- CONFIGURACIÓN GLOBAL Y PARÁMETROS ---
# Valores por defecto (se pueden sobrescribir por CLI)
DEFAULT_VIDEO = r"./input_videos/video2_corto.mp4"
FRAME_SKIP = 1
MOVEMENT_THRESHOLD_AREA = 800 
QC_THRESHOLD = 60

# --- INICIALIZACIÓN DE OBJETOS PERMANENTES ---
FGBG_MODEL = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=True)
CLAHE_MODEL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def correr_analisis_completo(video_path, out_path=None, fgbg_model=None, clahe_model=None,
                             frame_skip=3, threshold_area=800, show_window=False):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return False

    # Preparar writer si se solicita salida
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # leer primer frame para dimensiones y ROI inicial
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
        return False

    height, width = initial_frame.shape[:2]
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps / max(1, frame_skip), (width, height))
        print(f"Guardando salida en: {out_path}")

    frame_count = 0
    start_time = time.time()
    log_data = []

    # ROI inicial
    roi_w, roi_h = 50, 50
    original_roi = (width // 2 - roi_w // 2, height // 2 - roi_h // 2, roi_w, roi_h)
    current_roi = original_roi
    dx_history, dy_history = [], []

    # volver al inicio del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            base_frame = frame.copy()
            
            # Estabilidad
            frame_stability, current_roi, stability_shift = process_frame_stability(
                initial_frame, base_frame, current_roi, original_roi, dx_history, dy_history
            )

            # QC
            frame_qc, qc_log = process_qc_frame(frame_stability, frame_count, threshold=QC_THRESHOLD)         
            
            # Movimiento
            frame_movement = detectar_movimiento_frame(frame_qc, fgbg_model, clahe_model, threshold_area)

            # Log básico
            if qc_log:
                entry = (
                f"[ALERTA] Frame {frame_count} | "
                f"QC={qc_log['qc']:.1f} | Blur={qc_log['blur']:.1f} | "
                f"Occ={qc_log['occ']:.1f} | Light={qc_log['light']:.1f} | "
                f"ALERTA={','.join(qc_log['failed'])}"
                )
                log_data.append(entry)

            # Mostrar o guardar
            if show_window:
                cv2.imshow('Análisis de Video Pipeline Completo', frame_movement)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer is not None:
                writer.write(frame_movement)

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        end_time = time.time()
        print(f"Procesamiento finalizado. Frames procesados: {frame_count}. Tiempo total: {end_time - start_time:.2f}s")
    return log_data

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de análisis de video (QC, Estabilidad, Movimiento).")
    parser.add_argument("--video", "-v", default=DEFAULT_VIDEO, help="Ruta al video de entrada (relativa a /app en contenedor).")
    parser.add_argument("--headless", action="store_true", help="Ejecutar en modo headless (no abrir ventanas).")
    parser.add_argument("--out", "-o", default="./outputs/output_result.mp4", help="Ruta de video de salida (opcional).")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP, help="Saltar frames para acelerar.")
    parser.add_argument("--threshold-area", type=int, default=MOVEMENT_THRESHOLD_AREA, help="Umbral area movimiento.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # asegurar carpetas
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    success = correr_analisis_completo(
        video_path=args.video,
        out_path=args.out if args.headless else None,
        fgbg_model=FGBG_MODEL,
        clahe_model=CLAHE_MODEL,
        frame_skip=args.frame_skip,
        threshold_area=args.threshold_area,
        show_window=not args.headless
    )
    if not success:
        exit(1)
