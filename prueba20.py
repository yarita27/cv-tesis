
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import queue
import json
import csv
import torch
import os
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import ffmpeg
from collections import Counter

# Carpeta para guardar los recortes de vehículos
output_dir = "vehiculos_recortados"
os.makedirs(output_dir, exist_ok=True)
saved_crops = set()


# Activar optimizaciones de GPU para PyTorch
torch.backends.cudnn.benchmark = True

# Inicialización de variables globales
roi_points_1 = []
roi_points_2 = []
selecting_roi = 1
original_frame_shape = None
line_start = []
line_end = []
vehicle_times = {}
vehicle_speeds = {}
speed_distance_m = 3
vehicle_colors = {}  # track_id -> list of color names
vehicle_last_roi = {}         # track_id -> último ROI válido
previous_centers = {}  # track_id -> (cx, cy)
vehicle_roi_state = {}  # track_id -> "ROI1", "ROI2", o "fuera"

vehicle_final_info = {}  # track_id -> info completa final


class_counts_roi1 = {cls: set() for cls in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'ambulance',
                                            'person', 'child', 'adult', 'elderly', 'wheelchair', 'crutch', 'walking frame']}
class_counts_roi2 = {cls: set() for cls in class_counts_roi1.keys()}

frame_count_global = 0
already_saved = set()  # track_id ya procesados
track_id_to_bbox = {}  # track_id -> última posición conocida (x1, y1, x2, y2)

vehicle_metadata = {}  # track_id -> {"color": ..., "roi": ...}
vehicle_color_samples = {}  # track_id -> lista de LAB dominantes



vehicle_positions = {}  # Diccionario para almacenar posiciones históricas por track_id
PERSPECTIVE_SCALE = 0.05  # Valor inicial, será calibrado

# REEMPLAZA todo el sistema LAB con este enfoque HSV más robusto
VEHICLE_COLORS_HSV_RANGES = {
    "rojo": ([0, 100, 100], [10, 255, 255]),
    "azul": ([100, 100, 100], [140, 255, 255]),
    "verde": ([40, 100, 100], [80, 255, 255]),
    "amarillo": ([20, 100, 100], [40, 255, 255]),
    "blanco": ([0, 0, 200], [180, 50, 255]),
    "negro": ([0, 0, 0], [180, 255, 50]),
    "gris": ([0, 0, 50], [180, 50, 200]),
    "beige": ([20, 50, 150], [40, 150, 255]),
    "plateado": ([0, 0, 100], [180, 30, 200])
}

def classify_vehicle_color_hsv(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        max_pixels = 0
        dominant_color = "desconocido"
        
        for color, (lower, upper) in VEHICLE_COLORS_HSV_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = np.count_nonzero(mask)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color
        
        return dominant_color if max_pixels > (img.size * 0.05) else "desconocido"
    except:
        return "desconocido"
    
def cruzo_linea(p1, p2, prev_point, curr_point):
    """Verifica si la línea entre prev_point y curr_point cruza la línea p1-p2"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    A, B = p1, p2
    C, D = prev_point, curr_point
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def is_in_roi(bbox, roi_points):
    x1, y1, x2, y2 = bbox
    roi_poly = np.array(roi_points)

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points) - [x1, y1]], 255)

    # Recorta el área del bbox
    roi_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for y in range(y2 - y1):
        for x in range(x2 - x1):
            if 0 <= y1 + y < original_frame_shape[0] and 0 <= x1 + x < original_frame_shape[1]:
                if cv2.pointPolygonTest(roi_poly, (x1 + x, y1 + y), False) >= 0:
                    roi_crop[y, x] = 255

    area_total = (x2 - x1) * (y2 - y1)
    area_in = np.count_nonzero(roi_crop)

    return area_in / area_total > 0.2  # ➕ Tolerancia ajustable


VEHICLE_COLORS_LAB_RANGES = {
    "blanco": {
        "L": (120, 255),
        "a": (120, 135),
        "b": (115, 130)
    },
    "gris": {
        "L": (60, 120),
        "a": (120, 135),
        "b": (115, 130)
    },
    "negro": {
        "L": (0, 60),
        "a": (120, 135),
        "b": (115, 130)
    }
}

vehicle_lab_samples = {}  # track_id -> list of (L, a, b)

VEHICLE_COLORS_LAB_CENTERS = {
    "azul": (50, 140, 80),
    "rojo": (55, 175, 150),
    "verde": (60, 85, 110),
    "gris": (125, 128, 128),
    "blanco": (240, 128, 128),
    "negro": (20, 128, 128),
    "beige": (190, 125, 135)
}
LAB_DISTANCE_THRESHOLD = 25

def classify_color_by_distance(L, a, b):
    min_distance = float('inf')
    best_color = "desconocido"
    for color_name, (Lc, ac, bc) in VEHICLE_COLORS_LAB_CENTERS.items():
        distance = np.sqrt((L - Lc)**2 + (a - ac)**2 + (b - bc)**2)
        if distance < min_distance:
            min_distance = distance
            best_color = color_name
    if min_distance <= LAB_DISTANCE_THRESHOLD:
        return best_color
    else:
        return "desconocido"

def preprocess_vehicle_simple(img):
    # Suavizado básico
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred

def classify_vehicle_color_with_ranges(img):
    try:
        img_clean = preprocess_vehicle_simple(img)
        dominant_lab = get_dominant_color_lab_kmeans(img_clean)

        avg_L, avg_a, avg_b = dominant_lab
        print(f"[DEBUG] Dominante LAB: L={avg_L:.1f}, a={avg_a:.1f}, b={avg_b:.1f}")

        for color_name, ranges in VEHICLE_COLORS_LAB_RANGES.items():
            if (ranges["L"][0] <= avg_L <= ranges["L"][1] and
                ranges["a"][0] <= avg_a <= ranges["a"][1] and
                ranges["b"][0] <= avg_b <= ranges["b"][1]):
                return color_name

        return "desconocido"

    except Exception as e:
        print(f"[ERROR] classify_vehicle_color_with_ranges: {e}")
        return "desconocido"

def classify_vehicle_color_kmeans(img, k=3):
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    img_flat = img_blur.reshape((-1, 3))

    # Filtra píxeles oscuros y muy claros
    mask_bgr = np.all(img_flat > 30, axis=1) & np.all(img_flat < 220, axis=1)
    img_flat = img_flat[mask_bgr]
    if img_flat.shape[0] < 50:
        return "desconocido"

    # Convertir a LAB
    img_lab = cv2.cvtColor(img_flat.reshape((-1,1,3)), cv2.COLOR_BGR2LAB).reshape((-1,3))
    
    # Filtra grises
    mask_color = (np.abs(img_lab[:,1] - 128) > 5) | (np.abs(img_lab[:,2] - 128) > 5)
    img_lab = img_lab[mask_color]
    if img_lab.shape[0] < 50:
        return "desconocido"

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img_lab)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Cluster más grande
    counts = np.bincount(labels)
    dominant_lab = centers[np.argmax(counts)]

    L, a, b = dominant_lab
    return classify_color_by_distance(L, a, b)

def classify_color_lab_ranges(L, a, b):
    for color_name, ranges in VEHICLE_COLORS_LAB_RANGES.items():
        if (ranges["L"][0] <= L <= ranges["L"][1] and
            ranges["a"][0] <= a <= ranges["a"][1] and
            ranges["b"][0] <= b <= ranges["b"][1]):
            return color_name
    return "desconocido"


def get_dominant_color_lab_kmeans(img, k=3):
    img = img.copy()
    img = img.reshape((-1, 3))

    # Filtrado opcional: descarta píxeles muy oscuros o muy claros
    mask = np.all(img > 30, axis=1) & np.all(img < 220, axis=1)
    img = img[mask]

    if img.shape[0] < 10:
        raise ValueError("Muy pocos píxeles útiles para clasificar color")

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    counts = np.bincount(labels)
    dominant = cluster_centers[np.argmax(counts)].astype(np.uint8)

    # Convierte a LAB
    dominant_bgr = np.uint8([[dominant]])
    dominant_lab = cv2.cvtColor(dominant_bgr, cv2.COLOR_BGR2LAB)[0][0]
    return dominant_lab


def obtener_hora_grabacion(video_path):
    try:
        metadata = ffmpeg.probe(video_path)
        creation_str = metadata.get('format', {}).get('tags', {}).get('creation_time', '')
        if creation_str:
            # Ajuste de UTC a Venezuela (UTC-4)
            creation_utc = datetime.strptime(creation_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            creation_local = creation_utc - timedelta(hours=4)
            print("📅 Hora de grabación detectada:", creation_local)
            return creation_local
        else:
            print("⚠️ No se detectó metadato de hora.")
            return None
    except Exception as e:
        print("❌ Error al obtener hora de grabación:", e)
        return None

def check_exit_and_save(track_id, roi_tag, frame, label, pos_msec):
    if track_id in already_saved:
        return

    bbox = list(map(int, track_id_to_bbox.get(track_id, (0, 0, 0, 0))))
    if roi_tag == "ROI1":
        still_in_roi = is_in_roi(bbox, roi_points_1)
    elif roi_tag == "ROI2":
        still_in_roi = is_in_roi(bbox, roi_points_2)
    else:
        still_in_roi = False

    if not still_in_roi:
        # Asegúrate de que haya datos de velocidad
        if track_id in vehicle_speeds:
            event_time = video_start_time + timedelta(milliseconds=pos_msec)
            timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
            vehicle_final_info[track_id] = {
                "tipo": label,
                "roi": roi_tag,
                "velocidad": vehicle_speeds[track_id]["velocidad"],
                "color": vehicle_colors.get(track_id, "desconocido"),
                "timestamp": timestamp_str
            }
            already_saved.add(track_id)
            print(f"[🟢 Final guardado tras salida del ROI] ID: {track_id}, velocidad: {vehicle_speeds[track_id]['velocidad']}")

def select_roi(event, x, y, flags, param):
    global roi_points_1, roi_points_2, selecting_roi, frame_roi, line_start, line_end
    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting_roi == 1:
            roi_points_1.append((x, y))
            if len(roi_points_1) > 1:
                cv2.line(frame_roi, roi_points_1[-2], roi_points_1[-1], (0, 255, 0), 2)
            cv2.circle(frame_roi, (x, y), 5, (0, 0, 255), -1)
        elif selecting_roi == 2:
            roi_points_2.append((x, y))
            if len(roi_points_2) > 1:
                cv2.line(frame_roi, roi_points_2[-2], roi_points_2[-1], (255, 0, 0), 2)
            cv2.circle(frame_roi, (x, y), 5, (0, 255, 255), -1)
        elif selecting_roi == 3 and len(line_start) < 2:
            line_start.append((x, y))
            cv2.circle(frame_roi, (x, y), 5, (255, 255, 0), -1)
            if len(line_start) == 2:
                cv2.line(frame_roi, line_start[0], line_start[1], (255, 255, 0), 2)
        elif selecting_roi == 4 and len(line_end) < 2:
            line_end.append((x, y))
            cv2.circle(frame_roi, (x, y), 5, (0, 255, 255), -1)
            if len(line_end) == 2:
                cv2.line(frame_roi, line_end[0], line_end[1], (0, 255, 255), 2)
        cv2.imshow("Selecciona los ROIs", frame_roi)

def define_rois(video_source):
    global frame_roi, roi_points_1, roi_points_2, selecting_roi
    cap = cv2.VideoCapture(video_source)
    ret, frame = cap.read()

    if not ret:
        print("Error al leer el video.")
        return None

    frame_roi = frame.copy()
    cv2.namedWindow("Selecciona los ROIs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Selecciona los ROIs", frame.shape[1], frame.shape[0])
    cv2.imshow("Selecciona los ROIs", frame_roi)
    cv2.setMouseCallback("Selecciona los ROIs", select_roi)

    print("Selecciona ROI 1 (verde). Presiona ENTER cuando termines.")
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break
    selecting_roi = 2
    print("Selecciona ROI 2 (azul). Presiona ENTER cuando termines.")
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break

    selecting_roi = 3
    print("Selecciona dos puntos para la línea de inicio (amarillo). Presiona ENTER cuando termines.")
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break

    selecting_roi = 4
    print("Selecciona dos puntos para la línea de fin (celeste). Presiona ENTER cuando termines.")
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(roi_points_1) < 3 or len(roi_points_2) < 3:
        print("Cada ROI necesita al menos 3 puntos.")
        return None
    roi_points_1.append(roi_points_1[0])
    roi_points_2.append(roi_points_2[0])

    print(f"[DEBUG] Línea de inicio: {line_start}")
    print(f"[DEBUG] Línea de fin: {line_end}")

    return frame.shape[:2], video_start_time

def get_roi_mask(frame, roi_points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points)], 255)
    return mask

model_vehicles = YOLO('yolov8m.pt').to('cuda')
model_pedestrians = YOLO('yolov8n.pt').to('cuda')
deep_sort = DeepSort(max_age=10, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=None)

video_path = 'videocamara1_procesado2.mp4'
video_start_time = obtener_hora_grabacion(video_path)

if video_start_time is None:
    # Si no se puede obtener, usa hora de ejecución como fallback
    video_start_time = datetime.now()


video_source = 'videocamara1_procesado2.mp4'

result = define_rois(video_source)
if result is None:
    exit()
original_frame_shape, video_start_time = result


if original_frame_shape is None:
    exit()

def capture_frames(video_source, raw_queue, stop_event):
    cap = cv2.VideoCapture(video_source)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)  # ⏱️ Nuevo: tiempo actual del video

        if frame.shape[:2] != original_frame_shape:
            frame = cv2.resize(frame, (original_frame_shape[1], original_frame_shape[0]))

        try:
            raw_queue.put((frame, pos_msec), timeout=0.5)  # ⬅️ Envías frame + tiempo
        except queue.Full:
            continue
    cap.release()

def detection_and_tracking(raw_queue, processed_queue, stop_event):
    global frame_count_global
    global video_start_time
    while not stop_event.is_set() or not raw_queue.empty():
        try:
            frame, pos_msec = raw_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        frame_count_global += 1

        live_roi1 = set()
        live_roi2 = set()

        roi_mask_1 = get_roi_mask(frame, roi_points_1)
        roi_mask_2 = get_roi_mask(frame, roi_points_2)

        detections = []
        results_vehicles = model_vehicles(frame)
        results_pedestrians = model_pedestrians(frame)

        for results, model in zip([results_vehicles, results_pedestrians], [model_vehicles, model_pedestrians]):
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Filtrado de tamaño
                    box_w = x2 - x1
                    box_h = y2 - y1
                    frame_h, frame_w = frame.shape[:2]

                    if box_w < frame_w * 0.02 or box_h < frame_h * 0.02:
                        continue  # Muy pequeño, ignorar
                    if box_w > frame_w * 0.8 or box_h > frame_h * 0.8:
                        continue  # Muy grande, ignorar

                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    confidence = float(box.conf[0])
                    if confidence < 0.4:
                        continue
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cls_name in class_counts_roi1:
                        if roi_mask_1[cy, cx] == 255:
                            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, cls_name+'_ROI1'))
                        elif roi_mask_2[cy, cx] == 255:
                            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, cls_name+'_ROI2'))

        tracks = deep_sort.update_tracks(detections, frame=frame)

        #roi1_counter = 0
        #roi2_counter = 0
        

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            det_class = track.get_det_class()
            label, roi_tag = det_class.split('_')
            vehicle_last_roi[track_id] = roi_tag
            track_id_to_bbox[track_id] = (x1, y1, x2, y2)

            # 🟢 Solo contar una vez por track_id
            #if track_id not in vehicle_final_info:
             #   if roi_tag == 'ROI1':
              #      roi1_counter += 1
               # elif roi_tag == 'ROI2':
                #    roi2_counter += 1

            # 👉 Nueva lógica: detectar si salió del ROI
            in_roi1 = is_in_roi([x1, y1, x2, y2], roi_points_1)
            in_roi2 = is_in_roi([x1, y1, x2, y2], roi_points_2)
            actual_roi = "ROI1" if in_roi1 else "ROI2" if in_roi2 else "fuera"

            if in_roi1:
                live_roi1.add(track_id)
            if in_roi2:
                live_roi2.add(track_id)

            prev_state = vehicle_roi_state.get(track_id, None)
            vehicle_roi_state[track_id] = actual_roi

            if prev_state in ("ROI1", "ROI2") and actual_roi == "fuera":
                if track_id in vehicle_speeds:
                    # Siempre actualizar la info final al salir del ROI
                    vehicle_final_info[track_id] = {
                        "tipo": label,
                        "roi": prev_state,
                        "velocidad": vehicle_speeds[track_id]["velocidad"],
                        "color": vehicle_colors.get(track_id, "desconocido"),
                        "timestamp": vehicle_speeds[track_id]["timestamp"]
                    }
                    print(f"[📤 Registro ACTUALIZADO tras salida] ID: {track_id}")



            # Calcular color siempre para ir promediando y refinando
            cropped = frame[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h > 10 and w > 10:
                crop_margin_h = int(h * 0.35)
                crop_margin_w = int(w * 0.35)
                inner = cropped[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w]
                if track_id not in saved_crops:
                    crop_filename = f"{output_dir}/vehiculo_{track_id}.jpg"
                    cv2.imwrite(crop_filename, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_crops.add(track_id)

                # REEMPLAZA el bloque de clasificación de color con este:
                if inner.size > 0:
                    color_name = classify_vehicle_color_hsv(inner)
                    
                    if track_id not in vehicle_colors:
                        vehicle_colors[track_id] = color_name
                    elif vehicle_colors[track_id] != color_name:
                        # Actualizar solo si la confianza es alta
                        vehicle_colors[track_id] = color_name
                    
                    cv2.putText(frame, f"Color: {color_name}", (x1, y2 + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Guardar el punto anterior
            curr_center = (cx, cy)
            prev_center = previous_centers.get(track_id, curr_center)
            previous_centers[track_id] = curr_center

            # Verificar cruce línea de inicio
            if len(line_start) == 2:
                if cruzo_linea(line_start[0], line_start[1], prev_center, curr_center):
                    if track_id not in vehicle_times:
                        vehicle_times[track_id] = {'start': pos_msec, 'end': None}
                        print(f"[🚦 START] Vehículo {track_id} cruzó línea de inicio en {pos_msec:.0f} ms")

            # Verificar cruce línea de fin
            if len(line_end) == 2:
                if track_id in vehicle_times and vehicle_times[track_id]['end'] is None:
                    if cruzo_linea(line_end[0], line_end[1], prev_center, curr_center):
                        vehicle_times[track_id]['end'] = pos_msec
                        event_time = video_start_time + timedelta(milliseconds=pos_msec)
                        timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
                        t1 = vehicle_times[track_id]['start']
                        t2 = vehicle_times[track_id]['end']
                        if t1 is not None:
                            elapsed = (t2 - t1) / 1000  # pasa a segundos
                            if elapsed > 0:
                                speed = (speed_distance_m / elapsed) * 3.6
                                vehicle_speeds[track_id] = {
                                    "velocidad": round(speed, 2),
                                    "timestamp": timestamp_str,
                                    "color": vehicle_colors.get(track_id, "desconocido")
                                }
                                vehicle_final_info[track_id] = {
                                    "tipo": label,
                                    "roi": roi_tag,
                                    "velocidad": round(speed, 2),
                                    "color": vehicle_colors.get(track_id, "desconocido"),
                                    "timestamp": timestamp_str
                                }
                                print(f"[🏁 END] Vehículo {track_id} - Tiempo: {elapsed:.0f} ms - Velocidad: {speed:.2f} km/h")

            if len(line_end) == 2:
                if track_id in vehicle_times and vehicle_times[track_id]['end'] is None:
                    if cruzo_linea(line_end[0], line_end[1], prev_center, curr_center):
                        cv2.putText(frame, f"END: {track_id}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        cv2.circle(frame, curr_center, 4, (0, 255, 255), -1)
                        cv2.circle(frame, prev_center, 4, (0, 100, 255), -1)

                        vehicle_times[track_id]['end'] = pos_msec
                        event_time = video_start_time + timedelta(milliseconds=pos_msec)
                        timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
                        t1 = vehicle_times[track_id]['start']
                        t2 = vehicle_times[track_id]['end']
                        elapsed = t2 - t1
                        if elapsed > 0:
                            speed = (speed_distance_m / elapsed) * 3.6
                            vehicle_speeds[track_id] = {
                                "velocidad": round(speed, 2),
                                "timestamp": timestamp_str,
                                "color": vehicle_colors.get(track_id, "desconocido")
                            }

                            print(f"[✔️ Registro guardado] ID: {track_id}, velocidad: {speed:.2f}, timestamp: {timestamp_str}")

            color = (0, 255, 0) if roi_tag == 'ROI1' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if track_id in vehicle_speeds:
                cv2.putText(frame, f"{vehicle_speeds[track_id]} km/h", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if track_id in vehicle_colors:
                    color_label = vehicle_colors[track_id]
                    cv2.putText(frame, f"{color_label}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Mostrar color y ROI inicial si ya está identificado
            if track_id in vehicle_metadata:
                meta = vehicle_metadata[track_id]
                color_text = meta.get("color", "Desconocido")
                roi_text = meta.get("roi", "Desconocido")
                cv2.putText(frame, f"Color: {color_text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"ROI: {roi_text}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            if track_id in vehicle_speeds:
                check_exit_and_save(track_id, roi_tag, frame, label, pos_msec)



        cv2.polylines(frame, [np.array(roi_points_1)], True, (0,255,0), 2)
        cv2.polylines(frame, [np.array(roi_points_2)], True, (255,0,0), 2)
        if len(line_start) == 2:
            cv2.line(frame, line_start[0], line_start[1], (255, 255, 0), 2)
        if len(line_end) == 2:
            cv2.line(frame, line_end[0], line_end[1], (0, 255, 255), 2)

        # 👇 Conteo en vivo de los presentes en este frame
        #cv2.putText(frame, f"ROI1 (en vivo): {roi1_counter}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        #cv2.putText(frame, f"ROI2 (en vivo): {roi2_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        cv2.putText(frame, f"ROI1 (en vivo): {len(live_roi1)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"ROI2 (en vivo): {len(live_roi2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        print("Fecha de grabación base:", video_start_time.strftime("%Y-%m-%d %H:%M:%S"))


        try:
            processed_queue.put(frame, timeout=0.5)
        except queue.Full:
            continue

raw_queue = queue.Queue(maxsize=5)
processed_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()

capture_thread = threading.Thread(target=capture_frames, args=(video_source, raw_queue, stop_event))
detection_thread = threading.Thread(target=detection_and_tracking, args=(raw_queue, processed_queue, stop_event))

capture_thread.start()
detection_thread.start()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", original_frame_shape[1], original_frame_shape[0])
try:
    while not stop_event.is_set():
        try:
            proc_frame = processed_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        cv2.imshow("Frame", proc_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
except KeyboardInterrupt:
    stop_event.set()

capture_thread.join()
detection_thread.join()
cv2.destroyAllWindows()

final_roi1_count = sum(1 for roi in vehicle_last_roi.values() if roi == "ROI1")
final_roi2_count = sum(1 for roi in vehicle_last_roi.values() if roi == "ROI2")

print("\n✅ Conteo FINAL (donde terminaron):")
print(f"ROI1: {final_roi1_count}")
print(f"ROI2: {final_roi2_count}")

print("\n📊 Vehículos con datos finales registrados:")
for track_id, info in vehicle_final_info.items():
    print(f"ID: {track_id}, Datos: {info}")

os.makedirs("resultados", exist_ok=True)
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"resultados/vehiculos_detectados_{fecha_actual}.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "Tipo", "ROI final", "Velocidad (km/h)", "Color", "Fecha y Hora"])
    for track_id, info in vehicle_final_info.items():
        writer.writerow([
            track_id,
            info.get("tipo", "desconocido"),
            info.get("roi", "desconocido"),
            info.get("velocidad", "N/A"),
            info.get("color", "desconocido"),
            info.get("timestamp", "desconocido")
        ])
