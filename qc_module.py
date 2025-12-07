import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import zipfile
import cv2

#Modulo 1 Diagnostico de imagenes
'''
·Implementar un algoritmo (clásico o Deep Learning ligero) que analice cada frame (o un 
muestreo) para detectar anomalías ópticas. 
● Requisitos: Detectar al los siguientes problemas: 
○ Blurring/Desenfoque: Pérdida de nitidez. 
○ Oclusión/Suciedad: Lente tapado, gotas de agua o polvo excesivo. 
○ Low Light/Glare: Iluminación insuficiente o destellos cegadores. 
● Output: Generar un QC_Score (0-100) en tiempo real sobre el video y un log de alertas si la 
calidad baja de un umbral (ej. < 60%), para cada categoria enunciada
'''

def calculate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_occlusion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    return np.mean(thresh == 255)

def calculate_low_light(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    overexposed_ratio = np.mean(gray > 240)
    return mean_brightness, overexposed_ratio

def qc_score(blurriness, occlusion, low_light):
    # blurriness normalization
    blurriness_score = np.interp(blurriness, [0, 450], [0, 100])

    # occlusion
    occlusion_score = (1 - occlusion) * 100

    # low light
    low_light_score = np.interp(low_light, [30, 150], [0, 100])

    overall = (
        ( blurriness_score +
        occlusion_score +
        low_light_score ) / 3
    )

    return overall, blurriness_score, occlusion_score, low_light_score

def process_qc_frame(frame, frame_count, threshold=60):
    blurriness = calculate_blurriness(frame)
    occlusion = calculate_occlusion(frame)
    low_light, overexposed = calculate_low_light(frame)

    overall, blur_score, occ_score, light_score = qc_score(
        blurriness, occlusion, low_light
    )

    cv2.putText(frame, f"QC Score: {overall:.1f}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0) if overall >= threshold else (0,0,255), 2)

    cv2.putText(frame, 
                f"Blur:{blur_score:.1f} Occ:{occ_score:.1f} Light:{light_score:.1f}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)

    failed = []
    if blur_score < threshold: failed.append("Blurriness")
    if occ_score < threshold: failed.append("Occlusion")
    if light_score < threshold: failed.append("LowLight")

    log_entry = None
    if failed:
        log_entry = {
            "frame": frame_count,
            "qc": overall,
            "blur": blur_score,
            "occ": occ_score,
            "light": light_score,
            "failed": failed
        }

    return frame, log_entry


def run_qc_video(video_path, threshold=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el video: {video_path}")

    frame_count = 0
    log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, log_entry = process_qc_frame(frame, frame_count, threshold)

        # guardar log
        if log_entry:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            entry = (
                f"[ALERTA] Frame {frame_count} (t={timestamp:.2f}s) | "
                f"QC={log_entry['qc']:.1f} | Blur={log_entry['blur']:.1f} | "
                f"Occ={log_entry['occ']:.1f} | Light={log_entry['light']:.1f} | "
                f"ALERTA={','.join(log_entry['failed'])}"
            )
            print(entry)
            log.append(entry)

        # mostrar
        cv2.imshow("QC Monitoring", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return log



# Ejecución
#video_path = r"F:\Diego\Descargas\delirio\video_2.mp4"
#video_path = r"F:\Diego\Descargas\delirio\video2_corto.mp4"
#video_path = r"F:\Diego\Descargas\delirio\24 (2025-05-12 00'00'34 - 2025-05-12 00'03'07).avi"
#video_path = r"F:\Diego\Descargas\delirio\24 (2025-05-12 08'22'13 - 2025-05-12 08'22'53).avi"
#video_path = r"F:\Diego\Descargas\delirio\video_1.mp4"
#log = run_qc_video(video_path, 60)