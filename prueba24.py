#Ahora limpiamos el codigo y trabajamos las infracciones
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading, queue
import csv, os, json, glob, subprocess
import torch
from datetime import datetime, timedelta
import ffmpeg
from collections import Counter
from semaforo_overlay import load_signals_config, SemaforoState, MQTTSemaforoBridge, draw_signal_overlays

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

# === VIDEO & DETECCIÓN ===
VIDEO_SOURCE = r"videos/videocamara1_procesado2.mp4" 
DETECTION_DOWNSCALE = 1.0  # 1.0 = sin downscale. Si luego necesitas velocidad, prueba 0.75 o 0.5

# Activar optimizaciones de GPU para PyTorch
torch.backends.cudnn.benchmark = True

# ROIs genéricos (se llenan después de dibujar)
rois = []               # lista de dicts {"id": "ROI1", "poly": [...], "mask": np.ndarray}


# Inicialización de variables globales
line_start = []
line_end = []
vehicle_times = {}
vehicle_speeds = {}
speed_distance_m = 3
vehicle_colors = {}  # track_id -> list of color names
previous_centers = {}  # track_id -> (cx, cy)
vehicle_roi_state = {}  # track_id -> "ROI1", "ROI2", o "fuera"
vehicle_final_info = {}  # track_id -> info completa final
already_saved = set()  # track_id ya procesados
vehicle_metadata = {}  # track_id -> {"color": ..., "roi": ...}

# --- STOP-LINE ---
stop_crossed = set()           # para no repetir el mismo cruce por track
on_stop_frames = {}            # track_id -> frames detenido en la línea
STOP_MIN_FRAMES = 6            # cuántos frames para considerar "bloqueo"

# --- CROSSWALK ---
on_crosswalk_frames = {}       # track_id -> frames detenido en la cebra
# ¿El centro del track estaba dentro de una cebra en el frame anterior?
prev_in_crosswalk = {}   # track_id -> bool
CROSSWALK_MIN_FRAMES = 8       # frames detenido para considerar bloqueo
CROSSWALK_STOP_PX = 1.5        # velocidad mínima (px/frame) para "detenido"

track_prev_center = {}         # track_id -> (cx, cy) del frame anterior

# Registro de eventos por track y por tipo
events_fired = {}        # track_id -> set({"STOPLINE_CROSS", "CROSSWALK_BLOCK", ...})
event_last_ms = {}       # (track_id, event_code) -> último ms (para cooldown)
EVENT_COOLDOWN_MS = 1500 # evita repetir EXACTAMENTE el mismo evento muy seguido



PALETTE = [(0,255,0),(255,0,0),(0,255,255),(255,0,255),(0,128,255),(255,128,0),(128,255,128)]
def roi_color(roi_id):
    try:
        idx = int(roi_id.replace("ROI","")) - 1
    except:
        idx = 0
    return PALETTE[idx % len(PALETTE)]

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

    # Filtro: si NO hay velocidad y estuvo muy poco dentro del ROI, no guardar
    if REQUIRE_MIN_FRAMES_IF_NO_SPEED and vehicle_speeds.get(track_id) is None:
        if frames_in_roi.get(track_id, 0) < MIN_FRAMES_IN_ROI:
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

def _video_id_from_path(path: str) -> str:
    base = os.path.basename(path); name, _ = os.path.splitext(base)
    return name

def _basename(path: str) -> str:
    return os.path.basename(path)

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_geometry_file_for_video(video_path: str, cfg_dir: str = "config_geom") -> str | None:
    """
    Reglas de búsqueda (en orden):
      1) geometry_<basename_del_video>.json
      2) cualquier *.json cuyo 'video_source' tenga mismo basename
         (si hay varios, el más reciente por 'timestamp' o mtime)
    """
    if not os.path.isdir(cfg_dir):
        return None
    vid_id = _video_id_from_path(video_path)

    # (1) por nombre
    exact = os.path.join(cfg_dir, f"geometry_{vid_id}.json")
    if os.path.exists(exact):
        return exact

    # (2) por video_source.basename
    candidates = []
    for p in glob.glob(os.path.join(cfg_dir, "*.json")):
        try:
            data = _load_json(p)
        except Exception:
            continue
        src = data.get("video_source", "")
        if _basename(src) == _basename(video_path):
            ts = data.get("timestamp")
            try:
                score = datetime.fromisoformat(ts) if ts else datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                score = datetime.fromtimestamp(os.path.getmtime(p))
            candidates.append((score, p))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None

def load_geometry_for_video(video_path: str, cfg_dir: str = "config_geom") -> dict | None:
    """Devuelve el dict del JSON (normalizado) o None."""
    path = _find_geometry_file_for_video(video_path, cfg_dir)
    if not path:
        return None
    try:
        data = _load_json(path)
        data.setdefault("rois", [])
        data.setdefault("crosswalks", [])
        data.setdefault("lines", {})
        data["lines"].setdefault("speed_AB", {})
        data["lines"]["speed_AB"].setdefault("A", [])
        data["lines"]["speed_AB"].setdefault("B", [])
        data["lines"].setdefault("stop_line", [])
        return data
    except Exception as e:
        print(f"[geom] Error leyendo {path}: {e}")
        return None

def try_run_define(video_path: str):
    """Intenta abrir tu definidor externo define.py y vuelve cuando cierre."""
    try:
        print("[geom] No hay JSON. Abriendo definidor (define.py)...")
        subprocess.run(["python", "define.py", "--video", video_path], check=False)
    except Exception as e:
        print(f"[geom] No pude lanzar define.py: {e}")


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

def side_of_line(pt, a, b):
    return (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])

def crossed_line(prev_pt, curr_pt, a, b):
    if prev_pt is None: return False
    s1 = side_of_line(prev_pt, a, b)
    s2 = side_of_line(curr_pt, a, b)
    return s1 * s2 < 0

def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1):
            inside = not inside
    return inside

def center_speed_px(curr_xy, prev_xy):
    if prev_xy is None: return 0.0
    dx = curr_xy[0]-prev_xy[0]; dy = curr_xy[1]-prev_xy[1]
    return (dx*dx + dy*dy) ** 0.5

def log_event_csv(path, timestamp_str, ms_video, track_id, tipo, nota):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    newf = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newf:
            w.writerow(["timestamp","ms_video","track_id","tipo","nota"])
        w.writerow([timestamp_str, int(ms_video), track_id, tipo, nota])

def _log_event_unified(timestamp_str, ms_video, track_id, roi, tipo, nota, img_path):
    newf = not os.path.exists(ALL_EVENTS_PATH)
    if newf:
        os.makedirs(os.path.dirname(ALL_EVENTS_PATH), exist_ok=True)
        with open(ALL_EVENTS_PATH, "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["timestamp","ms_video","track_id","roi","tipo","nota","img_path"])
    with open(ALL_EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow([timestamp_str, int(ms_video), track_id, roi, tipo, nota, img_path or ""])

def _save_event_snapshot(frame, x1, y1, x2, y2, prefix):
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        # intenta ampliar un poco si el bbox está degenerado
        pad = 10
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fname = f"{prefix}.jpg"
    fpath = os.path.join(IMG_DIR, fname)
    cv2.imwrite(fpath, crop)
    return fpath

def emit_event_once(track_id, event_code, pos_msec, timestamp_str, roi, frame, bbox_xyxy, note, frame_event_guard):
    """Dispara 1 sola vez por track (nunca se repite ese tipo en ese track). Devuelve True si se registró."""
    keyf = (track_id, event_code)
    if keyf in frame_event_guard:
        return False
    fired = events_fired.setdefault(track_id, set())
    if event_code in fired:
        return False
    # snapshot
    x1, y1, x2, y2 = bbox_xyxy
    img_path = _save_event_snapshot(frame, x1, y1, x2, y2, f"{event_code}_tid{track_id}_{int(pos_msec)}")
    _log_event_unified(timestamp_str, pos_msec, track_id, roi, event_code, note, img_path)
    fired.add(event_code)
    frame_event_guard.add(keyf)
    return True

def emit_event_cooldown(track_id, event_code, pos_msec, timestamp_str, roi, frame, bbox_xyxy, note, frame_event_guard, cooldown_ms=EVENT_COOLDOWN_MS):
    """Permite repetir el mismo tipo en un track, pero no más de 1 vez por ventana de cooldown."""
    keyf = (track_id, event_code)
    if keyf in frame_event_guard:
        return False
    key = (track_id, event_code)
    last = event_last_ms.get(key, -1e18)
    if pos_msec - last < cooldown_ms:
        return False
    event_last_ms[key] = pos_msec
    x1, y1, x2, y2 = bbox_xyxy
    img_path = _save_event_snapshot(frame, x1, y1, x2, y2, f"{event_code}_tid{track_id}_{int(pos_msec)}")
    _log_event_unified(timestamp_str, pos_msec, track_id, roi, event_code, note, img_path)
    events_fired.setdefault(track_id, set()).add(event_code)
    frame_event_guard.add(keyf)
    return True

def crosswalk_id_by_point(pt, polys):
    for i, poly in enumerate(polys, start=1):
        if point_in_polygon(pt, poly):
            return f"XW{i}"
    return None


#Carga de modelos YOLOV8
_device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
model_vehicles = YOLO('yolov8m.pt').to(_device)
model_pedestrians = YOLO('yolov8n.pt').to(_device)

deep_sort = DeepSort(max_age=10, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=None)

# === [MQTT-SETUP INICIO] ===
cfg = load_signals_config("signals_config.json")

# Estado compartido de los semáforos con “stale” a gris
semaforo_state = SemaforoState(stale_after_sec=cfg.get("stale_after_sec", 5.0))

# Snapshot inicial desde archivo (para que la UI no arranque “vacía”)
semaforo_state.bootstrap(cfg.get("bootstrap_state", {}))

# Hilo MQTT (actualiza en vivo)
mqtt_cfg = cfg["mqtt"]
signals_ids = [s["id"] for s in cfg["signals"]]
mqtt_thread = MQTTSemaforoBridge(
    host=mqtt_cfg["host"],
    port=mqtt_cfg.get("port", 1883),
    topic_prefix=mqtt_cfg.get("topic_prefix", "esp32/semaforos/"),
    signals_ids=signals_ids,
    state=semaforo_state,
    username=mqtt_cfg.get("username", ""),
    password=mqtt_cfg.get("password", "")
)
mqtt_thread.start()
# === [MQTT-SETUP FIN] ===


# Hora de grabación (si no hay metadato, usa ahora)
video_start_time = obtener_hora_grabacion(VIDEO_SOURCE) or datetime.now()

# ==== Identidad de esta ejecución y rutas de salida ====
BASE = os.path.splitext(os.path.basename(VIDEO_SOURCE))[0]
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("eventos", f"{BASE}_{RUN_TAG}")
IMG_DIR = os.path.join(RUN_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

ALL_EVENTS_PATH = os.path.join(RUN_DIR, "eventos.csv")
# Cabecera única
with open(ALL_EVENTS_PATH, "w", newline="", encoding="utf-8") as f:
    import csv
    w = csv.writer(f)
    w.writerow(["timestamp","ms_video","track_id","roi","tipo","nota","img_path"])

# Intentar cargar geometría desde JSON asociado al video
geo = load_geometry_for_video(VIDEO_SOURCE)

if not geo:
    # Si no hay JSON, intenta abrir tu definidor externo (define.py) para crearlo
    try_run_define(VIDEO_SOURCE)
    # e intenta cargar nuevamente
    geo = load_geometry_for_video(VIDEO_SOURCE)

if geo:
    print(f"[geom] Usando geometría desde JSON para '{os.path.basename(VIDEO_SOURCE)}'")
    # Tomamos primer frame solo para medidas
    cap_tmp = cv2.VideoCapture(VIDEO_SOURCE); ok, frame0 = cap_tmp.read(); cap_tmp.release()
    if not ok:
        raise SystemExit("No pude leer el primer frame del video.")
    FULL_H, FULL_W = frame0.shape[:2]

    # ROIs
    ROIS_POLYS = [[tuple(map(int, p)) for p in poly] for poly in geo["rois"]]
    rois = []
    for i, poly in enumerate(ROIS_POLYS, start=1):
        rois.append({
            "id": f"ROI{i}",
            "poly": poly,
            "mask": get_roi_mask_shape((FULL_H, FULL_W), poly)
        })
    # Cebras / Líneas / Distancia
    CROSSWALKS_POLYS = [[tuple(map(int, p)) for p in poly] for poly in geo.get("crosswalks", [])]
    A_raw = geo["lines"]["speed_AB"].get("A", [])
    B_raw = geo["lines"]["speed_AB"].get("B", [])
    line_start = [tuple(map(int, A_raw[0])), tuple(map(int, A_raw[1]))] if len(A_raw) == 2 else []
    line_end   = [tuple(map(int, B_raw[0])), tuple(map(int, B_raw[1]))] if len(B_raw) == 2 else []
    STOP_LINE_RAW = geo["lines"].get("stop_line", [])
    STOP_LINE = [tuple(map(int, STOP_LINE_RAW[0])), tuple(map(int, STOP_LINE_RAW[1]))] if len(STOP_LINE_RAW) == 2 else []
    speed_distance_m = float(geo.get("speed_distance_m", speed_distance_m))
else:
    # Si no hay JSON válido, evita NameError y sigue sin STOP/CEBRA
    CROSSWALKS_POLYS = []
    STOP_LINE = []


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
    global vehicle_roi_state, vehicle_speeds, vehicle_final_info, vehicle_colors
    global CROSSWALKS_POLYS, STOP_LINE, speed_distance_m, video_start_time
    global rois, roi_mask_by_id, FULL_H, FULL_W, output_dir, previous_centers

    while not stop_event.is_set() or not raw_queue.empty():
        try:
            frame, pos_msec = raw_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        frame_event_guard = set()   # guarda (track_id, event_code) ya emitidos en ESTE frame
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

            track_id = int(track.track_id)  # <- al inicio del loop por track

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Tiempo del video en ms (si no lo calculas antes en tu loop de frame)
            event_time = video_start_time + timedelta(milliseconds=pos_msec)
            timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')

            # ====== PASO 3: EVENTOS POR TRACK ======
            roi_for_event = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2) or "fuera"
            bbox_xyxy = (x1, y1, x2, y2)

            # --- STOP-LINE (cruce + bloqueo) ---
            if STOP_LINE and len(STOP_LINE) == 2:
                A, B = STOP_LINE
                prevc = track_prev_center.get(track_id)
                currc = (cx, cy)

                # 1) Cruce de línea (una sola vez por track)
                if crossed_line(prevc, currc, A, B):
                    if emit_event_once(track_id, "STOPLINE_CROSS", pos_msec, timestamp_str,
                                    roi_for_event, frame, bbox_xyxy, "Cruzó la línea de pare", frame_event_guard):
                        cv2.putText(frame, "STOPLINE_CROSS", (x1, max(20, y1-25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 2) Bloqueo (detenido sobre la línea)
                dist_px = abs(side_of_line(currc, A, B)) / ( ((B[0]-A[0])**2 + (B[1]-A[1])**2)**0.5 + 1e-9 )
                spd_px  = center_speed_px(currc, prevc)
                if dist_px <= 3.0 and spd_px <= 1.5:
                    cnt = on_stop_frames.get(track_id, 0) + 1
                    on_stop_frames[track_id] = cnt
                    if cnt == STOP_MIN_FRAMES:
                        if emit_event_cooldown(track_id, "STOPLINE_BLOCK", pos_msec, timestamp_str,
                                            roi_for_event, frame, bbox_xyxy,
                                            f"Detenido sobre la línea ≥{STOP_MIN_FRAMES} frames", frame_event_guard):
                            cv2.putText(frame, "STOPLINE_BLOCK", (x1, max(20, y1-25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                else:
                    on_stop_frames[track_id] = 0

            # --- CROSSWALK (bloqueo por detención en cebra) ---
            en_cebra = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS)
            prevc = track_prev_center.get(track_id)
            if en_cebra and center_speed_px((cx, cy), prevc) <= CROSSWALK_STOP_PX:
                cnt = on_crosswalk_frames.get(track_id, 0) + 1
                on_crosswalk_frames[track_id] = cnt
                if cnt == CROSSWALK_MIN_FRAMES:
                    if emit_event_cooldown(track_id, "CROSSWALK_BLOCK", pos_msec, timestamp_str,
                                        roi_for_event, frame, bbox_xyxy,
                                        f"Detenido en cebra ≥{CROSSWALK_MIN_FRAMES} frames", frame_event_guard):
                        cv2.putText(frame, "CROSSWALK_BLOCK", (x1, max(20, y1-25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                on_crosswalk_frames[track_id] = 0

            # ===== CRUCE (entrada) de CEBRA =====
            en_cebra = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS)
            was_in = prev_in_crosswalk.get(track_id, False)

            # ROI para el CSV (puede ser ROI del carril, o la cebra detectada)
            roi_for_event = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2) or "fuera"
            xw_id = crosswalk_id_by_point((cx, cy), CROSSWALKS_POLYS) or ""

            bbox_xyxy = (x1, y1, x2, y2)

            # 3.1 Entrada a cebra (solo transición False -> True)
            if en_cebra and not was_in:
                note = "Entró al paso peatonal"
                if xw_id:
                    note += f" ({xw_id})"
                emit_event_once(
                    track_id, "CROSSWALK_ENTER", pos_msec, timestamp_str,
                    roi_for_event, frame, bbox_xyxy, note, frame_event_guard
                )
                cv2.putText(frame, "CROSSWALK_ENTER", (x1, max(20, y1-25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # (Opcional) 3.2 Salida de cebra (True -> False)
            if (not en_cebra) and was_in:
                note = "Salió del paso peatonal"
                if xw_id:
                    note += f" ({xw_id})"
                # Si quieres permitir múltiples salidas por el mismo track en distintas cebras, usa cooldown:
                emit_event_cooldown(
                    track_id, "CROSSWALK_EXIT", pos_msec, timestamp_str,
                    roi_for_event, frame, bbox_xyxy, note, frame_event_guard, cooldown_ms=1000
                )
                cv2.putText(frame, "CROSSWALK_EXIT", (x1, max(20, y1-25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Actualiza estado de cebra para el próximo frame
            prev_in_crosswalk[track_id] = en_cebra



            # 3.3 SIEMPRE actualiza el centro al final
            track_prev_center[track_id] = (cx, cy)
            
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

            # Color (recorte interior para reducir fondo)
            cropped = frame[y1:y2, x1:x2]
            h, w = cropped.shape[:2]
            if h > 10 and w > 10:
                crop_margin_h = int(h * 0.35)
                crop_margin_w = int(w * 0.35)
                inner = cropped[crop_margin_h:h - crop_margin_h, crop_margin_w:w - crop_margin_w]

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

                    # Limpieza de estados auxiliares para ese track
                    track_prev_center.pop(tid, None)
                    on_crosswalk_frames.pop(tid, None)
                    on_stop_frames.pop(tid, None)
                    prev_in_crosswalk.pop(tid, None)

                    # Opcional: limpiar historial de eventos del track
                    events_fired.pop(tid, None)
                    for k in list(event_last_ms.keys()):
                        if k[0] == tid:
                            event_last_ms.pop(k, None)
                    


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

        # STOP-LINE (rojo)
        if STOP_LINE and len(STOP_LINE) == 2:
            cv2.line(frame, STOP_LINE[0], STOP_LINE[1], (0, 0, 255), 2)  # rojo BGR
            cv2.putText(frame, "STOP", STOP_LINE[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # CEBRAS (magenta, por ejemplo)
        for poly in CROSSWALKS_POLYS:
            if len(poly) >= 3:
                arr = np.array(poly, np.int32)
                cv2.polylines(frame, [arr], True, (255, 0, 255), 2)  # magenta BGR


        # === Dibujar círculos del semáforo ===
        draw_signal_overlays(frame, cfg, semaforo_state)

        try:
            processed_queue.put(frame, timeout=0.5)
        except queue.Full:
            continue

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

# Finalizar lo que quedó dentro del ROI al terminar el video
for tid, state in list(vehicle_roi_state.items()):
    if state != "fuera" and tid not in already_saved:
        when = last_seen_msec.get(tid, 0.0)
        finalize_track(tid, state, when_msec=when)

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
