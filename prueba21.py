#Ahora trabajamos bajo la prioridad de mejorar calidad de imagen
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
import argparse
import sys

try:
    ROI_COUNT = int(os.getenv("ROI_COUNT") or input("¿Cuántos ROIs quieres? (Enter=2): ") or "2")
except Exception:
    ROI_COUNT = 2

# Forzamos el valor global que usan select_roi / define_rois
N_ROIS = ROI_COUNT
# Reiniciamos estructuras de selección para que todo case con N_ROIS
selecting_stage = 1
rois_points = [[] for _ in range(N_ROIS)]
line_start, line_end = [], []

print(f"[OK] Usaré {N_ROIS} ROIs")

# ==== Guardado independiente de líneas ====
SAVE_ON_EXIT = True                # guarda en cuanto sale del ROI
SAVE_ON_DISAPPEAR_AFTER_MS = 1500  # si deja de verse por ≥1.5s, guarda

MIN_FRAMES_IN_ROI = 5
REQUIRE_MIN_FRAMES_IF_NO_SPEED = True

frames_in_roi = {}         # track_id -> #frames dentro de algún ROI
last_seen_msec = {}        # track_id -> último pos_msec visto
track_id_to_label = {}     # track_id -> último label ("car", "bus", etc.)

best_crop_area = {}  # track_id -> área máxima vista
best_crop_path = {}  # track_id -> ruta del mejor PNG guardado

color_counts = {}  # track_id -> Counter()
vehicle_colors_best = {}      # track_id -> color desde best crop

# Carpeta para guardar los recortes de vehículos
output_dir = "vehiculos_recortados"
os.makedirs(output_dir, exist_ok=True)
saved_crops = set()

# === VIDEO & DETECCIÓN ===
VIDEO_SOURCE = r"videos/videocamara1_procesado2.mp4" 
DETECTION_DOWNSCALE = 1.0  # 1.0 = sin downscale. Si luego necesitas velocidad, prueba 0.75 o 0.5

# Activar optimizaciones de GPU para PyTorch
torch.backends.cudnn.benchmark = True

# ROIs genéricos (se llenan después de dibujar)
rois = []               # lista de dicts {"id": "ROI1", "poly": [...], "mask": np.ndarray}
roi_mask_by_id = {}     # dict id -> mask (para accesos rápidos)


# Inicialización de variables globales
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
PALETTE = [(0,255,0),(255,0,0),(0,255,255),(255,0,255),(0,128,255),(255,128,0),(128,255,128)]
def roi_color(roi_id):
    try:
        idx = int(roi_id.replace("ROI","")) - 1
    except:
        idx = 0
    return PALETTE[idx % len(PALETTE)]

"""def classify_vehicle_color_hsv(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        max_pixels = 0
        dominant_color = "desconocido"
        for color, (lower, upper) in VEHICLE_COLORS_HSV_RANGES.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = np.count_nonzero(mask)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color
        return dominant_color if max_pixels > max(50, total * 0.05) else "desconocido"
    except:
        return "desconocido"""

def classify_vehicle_color_hsv(img_bgr, margin=0.35):
    """
    Clasifica color de vehículo con paleta reducida (Vzla):
    blanco, negro, plateado, gris, beige/dorado, azul, azul oscuro,
    verde, rojo, vinotinto, amarillo/naranja

    Reglas clave:
    - Recorte interior para reducir fondo.
    - Primero decide escala de grises (blanco/negro/gris/plateado).
    - Prioriza rojos (incluye vinotinto = rojo oscuro).
    - Agrupa por hue para verde/azul/amarillo-naranja.
    - Devuelve SIEMPRE una etiqueta de la paleta.
    """
    if img_bgr is None or img_bgr.size == 0:
        return "gris"

    h, w = img_bgr.shape[:2]
    if h < 6 or w < 6:
        return "gris"

    # Recorte interior (menos borde/fondo)
    mh = int(h * margin); mw = int(w * margin)
    inner = img_bgr[mh:h-mh, mw:w-mw] if (h - 2*mh > 0 and w - 2*mw > 0) else img_bgr

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.uint16)          # 0..179
    S = hsv[..., 1].astype(np.float32) / 255.0 # 0..1
    V = hsv[..., 2].astype(np.float32) / 255.0 # 0..1

    # Filtra píxeles útiles (evita sombras/negras puras)
    valid = (V >= 0.12)
    if np.count_nonzero(valid) < 80:
        # fallback robusto por brillo si hay muy pocos válidos
        v_med = float(np.median(V))
        if v_med >= 0.82: return "blanco"
        if v_med <= 0.20: return "negro"
        return "gris"

    Hv = H[valid]; Sv = S[valid]; Vv = V[valid]
    s_med = float(np.median(Sv))
    v_med = float(np.median(Vv))

    # ---------- Escala de grises primero ----------
    # Umbrales reducidos para agrupar plateado/gris y evitar falsos colores
    if s_med <= 0.16:
        if v_med >= 0.82:
            return "blanco"
        if v_med <= 0.20:
            return "negro"
        # gris vs plateado (plateado más brillante)
        return "plateado" if v_med >= 0.58 else "gris"

    # ---------- Detectar ROJO / VINOTINTO con prioridad ----------
    # rojo en OpenCV es wrap (0..10 y 170..179)
    red_mask = ((Hv <= 10) | (Hv >= 170)) & (Sv >= 0.20)
    red_pixels = int(np.count_nonzero(red_mask))
    color_pixels = int(np.count_nonzero(Sv >= 0.20))
    frac_red = red_pixels / max(1, color_pixels)

    if frac_red >= 0.22:  # porcentaje suficiente de rojos
        v_med_red = float(np.median(Vv[red_mask])) if red_pixels > 0 else v_med
        # vinotinto = rojo oscuro (brillo bajo pero saturado)
        if v_med_red <= 0.38:
            return "vinotinto"
        return "rojo"

    # ---------- Agrupar por hue para el resto ----------
    # Buckets ajustados para vehículos
    # (rango OpenCV HSV)
    bins = {
        "amarillo/naranja": ((11, 35),),    # naranja/amarillo taxis
        "verde":            ((36, 85),),
        "azul":             ((96, 130),),
    }

    mask_color = Sv >= 0.20
    if not np.any(mask_color):
        # casi sin saturación: vuelve a grises
        return "plateado" if v_med >= 0.58 else "gris"

    Hc = Hv[mask_color]
    counts = {k: 0 for k in bins}
    for name, ranges in bins.items():
        c = 0
        for lo, hi in ranges:
            c += int(((Hc >= lo) & (Hc <= hi)).sum())
        counts[name] = c

    # Elige bucket más votado
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    if counts[best] == 0:
        # Sin votos claros -> grises
        return "plateado" if v_med >= 0.58 else "gris"

    # Ajustes de tonalidad (oscuro/claro) solo donde interesa
    if best == "azul":
        # azul oscuro si el brillo es bajo (pero no negro)
        return "azul oscuro" if v_med <= 0.45 else "azul"

    # beige/dorado: baja saturación con hue ~amarillo y brillo alto
    if best == "amarillo/naranja" and s_med <= 0.30 and v_med >= 0.62:
        return "beige/dorado"

    # verde/amarillo-naranja normal
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rois", type=int, default=2, help="Cantidad de ROIs a dibujar")
    return p.parse_args()

ARGS = parse_args()
N_ROIS = max(1, ARGS.rois)  # al menos 1

def roi_by_point(cx, cy, rois, W, H):
    cx = max(0, min(W-1, cx))
    cy = max(0, min(H-1, cy))
    for r in rois:
        if r["mask"][cy, cx] == 255:
            return r["id"]
    return None

def roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2):
    for r in rois:
        if bbox_in_mask(r["mask"], x1, y1, x2, y2, min_ratio=min_ratio):
            return r["id"]
    return None



def run_yolo_scaled(frame_full, model, scale=1.0):
    h_full, w_full = frame_full.shape[:2]
    if scale == 1.0:
        img = frame_full
        sx = sy = 1.0
    else:
        w_small = int(w_full * scale); h_small = int(h_full * scale)
        img = cv2.resize(frame_full, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        sx = w_full / float(w_small); sy = h_full / float(h_small)
    results = model(img)
    boxes = []
    for r in results:
        for b in r.boxes:
            x1s, y1s, x2s, y2s = b.xyxy[0]
            x1 = int(x1s * sx); y1 = int(y1s * sy)
            x2 = int(x2s * sx); y2 = int(y2s * sy)
            conf = float(b.conf[0])
            cls_name = model.names[int(b.cls[0])]
            boxes.append((x1, y1, x2, y2, conf, cls_name))
    return boxes
    
def cruzo_linea(p1, p2, prev_point, curr_point):
    """Verifica si la línea entre prev_point y curr_point cruza la línea p1-p2"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    A, B = p1, p2
    C, D = prev_point, curr_point
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

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


def obtener_hora_grabacion(VIDEO_SOURCE):
    try:
        metadata = ffmpeg.probe(VIDEO_SOURCE)
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

def finalize_track(track_id, final_roi, when_msec):
    if track_id in already_saved:
        return

    label = track_id_to_label.get(track_id, "vehiculo")

    # color en vivo ya consolidado por Counter
    color_live = vehicle_colors.get(track_id, "desconocido")
    color_best = vehicle_colors_best.get(track_id)  # <- puede ser None

    # Si no hubo color_best aún pero tenemos best PNG, clasifícalo ahora
    if color_best is None and (track_id in best_crop_path):
        try:
            crop_img = cv2.imread(best_crop_path[track_id])
            if crop_img is not None:
                h, w = crop_img.shape[:2]
                mh = int(h * 0.35); mw = int(w * 0.35)
                inner = crop_img[mh:h-mh, mw:w-mw] if (h-2*mh > 0 and w-2*mw > 0) else crop_img
                det2 = classify_vehicle_color_hsv(inner) if inner.size > 0 else classify_vehicle_color_hsv(crop_img)
                if det2:
                    color_best = det2
                    vehicle_colors_best[track_id] = det2
        except Exception:
            pass

    # timestamp
    event_time = video_start_time + timedelta(milliseconds=when_msec)
    timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')

    vehicle_final_info[track_id] = {
        "tipo": label,
        "roi": final_roi if final_roi else "desconocido",
        "velocidad": vehicle_speeds.get(track_id, {}).get("velocidad"),
        "color": color_live,            # <- se mantiene para compatibilidad
        "color_best": color_best,       # <- NUEVO atributo
        "timestamp": timestamp_str
    }
    already_saved.add(track_id)

def update_best_crop_and_color(frame, bbox, track_id):
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    h, w = cropped.shape[:2]
    if h <= 10 or w <= 10:
        return

    mh = int(h * 0.35); mw = int(w * 0.35)
    inner = cropped[mh:h-mh, mw:w-mw] if (h-2*mh > 0 and w-2*mw > 0) else cropped

    # --- best crop ---
    area = (x2 - x1) * (y2 - y1)
    was_best = False
    if area > best_crop_area.get(track_id, 0):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"vehiculo_{track_id}_best.png")
        if cv2.imwrite(path, cropped):
            best_crop_area[track_id] = area
            best_crop_path[track_id] = path
            vehicle_metadata.setdefault(track_id, {})["best_crop"] = path
            was_best = True

    # --- color en vivo (suma 1) ---
    if inner.size > 0:
        detected_live = classify_vehicle_color_hsv(inner)
        if detected_live:
            cc = color_counts.setdefault(track_id, Counter())
            cc[detected_live] += 1
            vehicle_colors[track_id] = cc.most_common(1)[0][0]

    # === NUEVO: color_best desde el best crop (sólo cuando hay nuevo best) ===
    if was_best:
        inner_best = inner if inner.size > 0 else cropped
        detected_best = classify_vehicle_color_hsv(inner_best)
        if not detected_best and cropped.size > 0:
            detected_best = classify_vehicle_color_hsv(cropped)
        if detected_best:
            vehicle_colors_best[track_id] = detected_best


def select_roi(event, x, y, flags, param):
    """
    selecting_stage:
      1..N_ROIS -> sumar puntos al polígono del ROI i
      N_ROIS+1  -> línea de inicio (2 puntos)
      N_ROIS+2  -> línea de fin (2 puntos)
    Clic izquierdo: agrega punto
    """
    global rois_points, selecting_stage, frame_roi, line_start, line_end
    roi_count = len(rois_points) if rois_points else 2
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # paleta básica para dibujar cada ROI con color distinto
    palette = [(0,255,0),(255,0,0),(0,255,255),(255,0,255),(0,128,255),(255,128,0),(128,255,128)]

    # --- ROIs dinámicos ---
    if 1 <= selecting_stage <= roi_count:
        idx = selecting_stage - 1
        rois_points[idx].append((x, y))
        color = palette[idx % len(palette)]
        # unir con el punto anterior
        if len(rois_points[idx]) > 1:
            cv2.line(frame_roi, rois_points[idx][-2], rois_points[idx][-1], color, 2)
        cv2.circle(frame_roi, (x, y), 5, (0, 0, 255), -1)

    # --- Línea de inicio ---
    elif selecting_stage == roi_count + 1:
        if len(line_start) < 2:
            line_start.append((x, y))
            cv2.circle(frame_roi, (x, y), 5, (255, 255, 0), -1)
            if len(line_start) == 2:
                cv2.line(frame_roi, line_start[0], line_start[1], (255,255,0), 2)

    # --- Línea de fin ---
    elif selecting_stage == roi_count + 2:
        if len(line_end) < 2:
            line_end.append((x, y))
            cv2.circle(frame_roi, (x, y), 5, (0, 255, 255), -1)
            if len(line_end) == 2:
                cv2.line(frame_roi, line_end[0], line_end[1], (0,255,255), 2)

    cv2.imshow("Selecciona los ROIs", frame_roi)


def define_rois(video_source):
    """
    Dibuja N_ROIS polígonos (ROI 1..N), luego dos líneas (inicio/fin).
    Devuelve (alto, ancho) del primer frame, o None si falla.
    """
    global frame_roi, rois_points, line_start, line_end, selecting_stage
    roi_count = len(rois_points) if rois_points else 2
    print(f"[DEBUG] define_rois pedirá {roi_count} ROIs")
    # limpia por si venías de una ejecución anterior
    for i in range(roi_count):
        rois_points[i].clear()
    line_start.clear()
    line_end.clear()
    selecting_stage = 1

    # lee primer frame (usa el método que ya te funciona; aquí va la versión simple)
    cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Error al leer el video para ROIs.")
        return None

    frame_roi = frame.copy()  # SIN resize
    cv2.namedWindow("Selecciona los ROIs", cv2.WINDOW_NORMAL)
    cv2.imshow("Selecciona los ROIs", frame_roi)
    cv2.setMouseCallback("Selecciona los ROIs", select_roi)

    # --- Selección de ROIs 1..N ---
    for i in range(roi_count):
        print(f"Selecciona ROI {i+1}. Presiona ENTER cuando termines.")
        while True:
            k = cv2.waitKey(16) & 0xFF
            if k == 13:  # ENTER
                if len(rois_points[i]) >= 3:
                    # cerrar visualmente el polígono
                    color = (0,255,0)
                    cv2.line(frame_roi, rois_points[i][-1], rois_points[i][0], color, 2)
                    cv2.imshow("Selecciona los ROIs", frame_roi)
                    break
                else:
                    print(f"ROI {i+1}: agrega al menos 3 puntos antes de ENTER.")
            elif k == 27:  # ESC para cancelar todo
                cv2.destroyAllWindows()
                return None
        selecting_stage += 1  # pasa al siguiente ROI

    # --- Líneas ---
    print("Selecciona dos puntos para la línea de inicio (amarillo). Presiona ENTER cuando termines.")
    while True:
        k = cv2.waitKey(16) & 0xFF
        if k == 13:
            if len(line_start) == 2:
                break
            else:
                print("Línea de inicio: faltan puntos (2).")
        elif k == 27:
            cv2.destroyAllWindows(); return None
    selecting_stage += 1

    print("Selecciona dos puntos para la línea de fin (celeste). Presiona ENTER cuando termines.")
    while True:
        k = cv2.waitKey(16) & 0xFF
        if k == 13:
            if len(line_end) == 2:
                break
            else:
                print("Línea de fin: faltan puntos (2).")
        elif k == 27:
            cv2.destroyAllWindows(); return None

    cv2.destroyAllWindows()
    return frame.shape[:2]



def get_roi_mask(frame, roi_points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points)], 255)
    return mask

def get_roi_mask_shape(shape_hw, roi_points):
    # Crea una máscara binaria del ROI usando solo (alto, ancho), sin depender de un frame.
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(roi_points) >= 3:
        pts = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def bbox_in_mask(mask, x1, y1, x2, y2, min_ratio=0.2):
    # Recorta la bbox a los límites de la imagen
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mask.shape[1], x2); y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    sub = mask[y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    inside = int(np.count_nonzero(sub))
    return (inside / max(1, area)) > min_ratio


model_vehicles = YOLO('yolov8m.pt').to('cuda')
model_pedestrians = YOLO('yolov8n.pt').to('cuda')
deep_sort = DeepSort(max_age=10, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=None)


# Hora de grabación (si no hay metadato, usa ahora)
video_start_time = obtener_hora_grabacion(VIDEO_SOURCE) or datetime.now()

# Define ROIs en resolución nativa una sola vez
frame_shape = define_rois(VIDEO_SOURCE)
if frame_shape is None:
    raise SystemExit(1)
FULL_H, FULL_W = map(int, frame_shape)   # asegura enteros
original_frame_shape = (FULL_H, FULL_W)  # compatibilidad con código existente

# Construir objetos ROI genéricos a partir de rois_points
rois = []
for i, poly in enumerate(rois_points, start=1):
    rois.append({
        "id": f"ROI{i}",
        "poly": poly,
        "mask": get_roi_mask_shape((FULL_H, FULL_W), poly)
    })
roi_mask_by_id = {r["id"]: r["mask"] for r in rois}


def capture_frames(VIDEO_SOURCE, raw_queue, stop_event):
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)  # ⏱️ Nuevo: tiempo actual del video

        try:
            raw_queue.put((frame, pos_msec), timeout=0.5)  # ⬅️ Envías frame + tiempo
        except queue.Full:
            continue
    cap.release()

def detection_and_tracking(raw_queue, processed_queue, stop_event):
    global frame_count_global, previous_centers, vehicle_last_roi, track_id_to_bbox
    global vehicle_roi_state, vehicle_speeds, vehicle_final_info, vehicle_colors, video_start_time
    global rois, roi_mask_by_id, FULL_H, FULL_W, output_dir, saved_crops
    
    while not stop_event.is_set() or not raw_queue.empty():
        try:
            frame, pos_msec = raw_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        frame_count_global += 1

        live_counts = Counter()
        live_now = set()  # <- tracks presentes en este frame



        detections = []
        # Detección en FULL-RES (si luego usas run_yolo_scaled, sustitúyelo aquí)
        veh = run_yolo_scaled(frame, model_vehicles, DETECTION_DOWNSCALE)
        ped = run_yolo_scaled(frame, model_pedestrians, DETECTION_DOWNSCALE)
        detections = []
        for (x1, y1, x2, y2, confidence, cls_name) in veh + ped:
            frame_h, frame_w = frame.shape[:2]
            box_w = x2 - x1; box_h = y2 - y1
            if box_w < frame_w * 0.02 or box_h < frame_h * 0.02:
                continue
            if box_w > frame_w * 0.8 or box_h > frame_h * 0.8:
                continue
            if confidence < 0.4:
                continue
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            roi_id = roi_by_point(cx, cy, rois, FULL_W, FULL_H)
            if roi_id is None:
                continue
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, f"{cls_name}_{roi_id}"))



        tracks = deep_sort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            
            actual_roi = "fuera"

            # 2) Calcula el ROI real por bbox (genérico para N ROIs)
            _roi = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2)  # devuelve "ROI1"/"ROI2"/... o None
            if _roi is not None:
                actual_roi = _roi

            # 3) Obtén la clase detectada de forma segura (puede venir sin '_')
            det_class = track.get_det_class() or ""
            if "_" in det_class:
                label, roi_tag = det_class.rsplit("_", 1)
            else:
                label = det_class if det_class else "obj"
                roi_tag = actual_roi if actual_roi != "fuera" else "desconocido"

            track_id_to_label[track_id] = label
            live_now.add(track_id)
            last_seen_msec[track_id] = pos_msec

            # --- transición de estado ROI ---
            prev_state = vehicle_roi_state.get(track_id, "fuera")
            vehicle_roi_state[track_id] = actual_roi  # "ROIx" o "fuera"

            # contar en vivo
            if actual_roi != "fuera":
                live_counts[actual_roi] += 1
                frames_in_roi[track_id] = frames_in_roi.get(track_id, 0) + 1

            # guardado inmediato al SALIR del ROI (sin líneas)
            if SAVE_ON_EXIT and (prev_state != "fuera") and (actual_roi == "fuera"):
                finalize_track(track_id, prev_state, when_msec=pos_msec)

            vehicle_last_roi[track_id] = roi_tag
            track_id_to_bbox[track_id] = (x1, y1, x2, y2)

            # Color (recorte interior para reducir fondo)
            cropped = frame[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h > 10 and w > 10:
                crop_margin_h = int(h * 0.35)
                crop_margin_w = int(w * 0.35)
                inner = cropped[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w]

                # Guardado de crop (opcional, como ya lo tienes)
                """if track_id not in saved_crops:
                    # Construye la ruta en PNG (sin compresión con pérdidas)
                    crop_filename = os.path.join(output_dir, f"vehiculo_{track_id}.png")
                    ok = cv2.imwrite(crop_filename, cropped)
                    if ok:
                        saved_crops.add(track_id)"""
                
                # === Guardar SOLO el "mejor" crop (mayor área) ===
                area = (x2 - x1) * (y2 - y1)
                prev_best = best_crop_area.get(track_id, 0)
                if area > prev_best:
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, f"vehiculo_{track_id}_best.png")
                    ok = cv2.imwrite(path, cropped)
                    if ok:
                        best_crop_area[track_id] = area
                        best_crop_path[track_id] = path
                        vehicle_metadata.setdefault(track_id, {})["best_crop"] = path


                # === Color estable por mayoría ===
                if inner.size > 0:
                    detected = classify_vehicle_color_hsv(inner)
                    if detected:  # por si tu función devolviera None/"" en algún frame raro
                        cc = color_counts.setdefault(track_id, Counter())
                        cc[detected] += 1
                        mode_color = cc.most_common(1)[0][0]
                        vehicle_colors[track_id] = mode_color


            # START/END para velocidad
            curr_center = (cx, cy)
            prev_center = previous_centers.get(track_id, curr_center)
            previous_centers[track_id] = curr_center

            if len(line_start) == 2:
                if cruzo_linea(line_start[0], line_start[1], prev_center, curr_center):
                    if track_id not in vehicle_times:
                        vehicle_times[track_id] = {'start': pos_msec, 'end': None}
                        print(f"[🚦 START] Vehículo {track_id} cruzó línea de inicio en {pos_msec:.0f} ms")

            if len(line_end) == 2:
                if track_id in vehicle_times and vehicle_times[track_id]['end'] is None:
                    if cruzo_linea(line_end[0], line_end[1], prev_center, curr_center):
                        vehicle_times[track_id]['end'] = pos_msec
                        event_time = video_start_time + timedelta(milliseconds=pos_msec)
                        timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
                        t1 = vehicle_times[track_id]['start']
                        t2 = vehicle_times[track_id]['end']
                        if t1 is not None:
                            elapsed = (t2 - t1) / 1000.0  # segundos
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
                                print(f"[🏁 END] Vehículo {track_id} - Tiempo: {elapsed:.2f} s - Velocidad: {speed:.2f} km/h")
                        cv2.putText(frame, f"END: {track_id}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            # Overlay
            # === Overlay (una sola vez, basado en estado en vivo) ===
            rect_roi = roi_tag if str(roi_tag).startswith("ROI") else (actual_roi if actual_roi != "fuera" else None)
            color_rect = roi_color(rect_roi) if rect_roi else (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)

            # --- NUEVO: colores vivo y best (en ese orden) ---
            display_color_live = vehicle_colors.get(track_id)
            display_color_best = vehicle_colors_best.get(track_id)   # <-- asegúrate de tener este dict global

            y_text = y2 + 15
            if display_color_live:
                cv2.putText(frame, f"Color(vivo): {display_color_live}", (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_text += 20

            if display_color_best:
                cv2.putText(frame, f"Color(best): {display_color_best}", (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)
                y_text += 20

            # ROI visible: si está fuera, muestra el último dentro
            display_roi = actual_roi if actual_roi != "fuera" else (prev_state if prev_state and prev_state != "fuera" else None)
            if display_roi:
                cv2.putText(frame, f"ROI: {display_roi}", (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
                y_text += 20

            # Velocidad (si existe)
            if track_id in vehicle_speeds:
                cv2.putText(frame, f"{vehicle_speeds[track_id]['velocidad']} km/h", (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


        # Cerrar tracks que desaparecieron sin cruzar fuera (p. ej. se perdió el track)
        if SAVE_ON_DISAPPEAR_AFTER_MS is not None:
            for tid, state in list(vehicle_roi_state.items()):
                if tid in live_now:
                    continue  # sigue activo
                if state == "fuera":
                    continue  # ya estaba fuera

                last_seen = last_seen_msec.get(tid, None)
                if last_seen is None:
                    continue
                if (pos_msec - last_seen) >= SAVE_ON_DISAPPEAR_AFTER_MS:
                    finalize_track(tid, state, when_msec=last_seen)
                    # Opcional: marcarlo como 'fuera' para no repetir
                    vehicle_roi_state[tid] = "fuera"


        # ROIs
        for i, r in enumerate(rois, start=1):
            color = (0,255,0)  # si quieres, asigna paleta por i
            cv2.polylines(frame, [np.array(r["poly"], dtype=np.int32)], True, color, 2)

        # Líneas
        if len(line_start) == 2:
            cv2.line(frame, line_start[0], line_start[1], (255, 255, 0), 2)
        if len(line_end) == 2:
            cv2.line(frame, line_end[0], line_end[1], (0, 255, 255), 2)

        # Conteo en vivo por ROI
        y = 110
        for roi_id, cnt in live_counts.items():
            cv2.putText(frame, f"{roi_id} (en vivo): {cnt}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            y += 40


        # Conteo en vivo
        #cv2.putText(frame, f"ROI1 (en vivo): {len(live_roi1)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        #cv2.putText(frame, f"ROI2 (en vivo): {len(live_roi2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        try:
            processed_queue.put(frame, timeout=0.5)
        except queue.Full:
            continue

# Finalizar lo que quedó dentro del ROI al terminar el video
for tid, state in list(vehicle_roi_state.items()):
    if state != "fuera" and tid not in already_saved:
        when = last_seen_msec.get(tid, 0.0)
        finalize_track(tid, state, when_msec=when)

raw_queue = queue.Queue(maxsize=5)
processed_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()

capture_thread = threading.Thread(target=capture_frames, args=(VIDEO_SOURCE, raw_queue, stop_event))
detection_thread = threading.Thread(target=detection_and_tracking, args=(raw_queue, processed_queue, stop_event))

capture_thread.start()
detection_thread.start()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

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

from collections import Counter
print("\n✅ Conteo FINAL (donde terminaron):")
final_counts = Counter()
for info in vehicle_final_info.values():
    rid = info.get("roi", "desconocido")
    final_counts[rid] += 1
# imprime en orden ROI1..ROIN si existen
for r in rois:
    rid = r["id"]
    print(f"{rid}: {final_counts.get(rid, 0)}")
# y cualquier otro (p.ej. 'desconocido')
for k, v in final_counts.items():
    if k not in [r["id"] for r in rois]:
        print(f"{k}: {v}")


print("\n📊 Vehículos con datos finales registrados:")
for track_id, info in vehicle_final_info.items():
    print(f"ID: {track_id}, Datos: {info}")

os.makedirs("resultados", exist_ok=True)
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f"resultados/vehiculos_detectados_{fecha_actual}.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    # Encabezados con color_best y best_crop
    writer.writerow([
        "Track ID", "Tipo", "ROI final", "Velocidad (km/h)",
        "Color (vivo)", "Color (best)", "Best Crop", "Fecha y Hora"
    ])
    for track_id, info in vehicle_final_info.items():
        writer.writerow([
            track_id,
            info.get("tipo", "desconocido"),
            info.get("roi", "desconocido"),
            info.get("velocidad", "N/A"),
            info.get("color", "desconocido"),  # color en vivo
            info.get("color_best", vehicle_colors_best.get(track_id, "desconocido")),
            vehicle_metadata.get(track_id, {}).get("best_crop"),
            info.get("timestamp", "desconocido")
        ])
