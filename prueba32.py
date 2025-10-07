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
from semaforo_overlay import (
    load_signals_config,
    SemaforoState,
    SSHSemaforoBridge,
    draw_signal_overlays
)


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
VIDEO_SOURCE = r"videos/velocidad_peaton_procesado.mp4" 
#videos/lluvialunes_procesado.mp4
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
on_stop_frames = {}            # track_id -> frames detenido en la línea
STOP_MIN_FRAMES = 6            # cuántos frames para considerar "bloqueo"

# --- CROSSWALK ---
on_crosswalk_frames = {}       # track_id -> frames detenido en el paso peatonal
# ¿El centro del track estaba dentro de un paso peatonal en el frame anterior?
prev_in_crosswalk = {}   # track_id -> bool
CROSSWALK_MIN_FRAMES = 8       # frames detenido para considerar bloqueo
CROSSWALK_STOP_PX = 1.5        # velocidad mínima (px/frame) para "detenido"

track_prev_center = {}         # track_id -> (cx, cy) del frame anterior

# Control de infracciones por track y por tipo
infra_seen = set()             # {(track_id, infr_code)} -> ya registrada 1 vez
infra_last_ms = {}             # {(track_id, infr_code)} -> último ms (cooldown)

INFRA_COOLDOWN_MS = 1500       # 1.5s típico, ajusta por tipo si quieres

# Umbral de prueba (3 km/h, cámbialo en producción)
SPEED_LIMIT_KMH = 20

# ====== Contra-sentido (config + estado) ======
WRONGWAY_MAX_MS = 15000   # ventana máx. entre cruces B→A para confirmarlo (15s)
MIN_PROJ_PX = 1.5         # avance mínimo hacia A en px por frame para considerar "va hacia A" antes 4.0; prueba 1.0–2.0 si sigue sin disparar

wrongway_prog = {}        # track_id -> {"first":"END"/"START", "t_first":ms}
AB_UNIT = None

# --- PEATONES: cruce fuera del paso ---
PEDESTRIAN_CROSS_MIN_FRAMES = 10     # mínimo de frames dentro de la calzada para considerarlo “cruce”
PEDESTRIAN_CROSS_MIN_PX = 40         # distancia mín. en píxeles recorrida sobre calzada (evita falsos)
# Sesión de peatón “en calzada” SIN paso peatonal
ped_session = {}  # track_id -> {"start_pos":(x,y), "start_ms":ms, "frames":int, "ever_in_paso":bool}
# Para saber si estaba en calzada en el frame anterior
prev_in_road_person = {}  # track_id -> bool
ped_speed = {}                 # track_id -> {"start_proj":float, "start_ms":ms}
prev_in_paso_person = {}       # track_id -> bool
# Velocidad de peatones en el paso
ped_in_paso_ms = {}         # track_id -> ms de entrada al paso
track_speed_kmh = {}        # track_id -> velocidad peatonal calculada (km/h)



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


    label = track_id_to_label.get(track_id, "vehiculo")
    has_veh_speed = (vehicle_speeds.get(track_id) is not None)
    has_ped_speed = (track_speed_kmh.get(track_id) is not None)
    if REQUIRE_MIN_FRAMES_IF_NO_SPEED and not (has_veh_speed or has_ped_speed):
        if frames_in_roi.get(track_id, 0) < MIN_FRAMES_IN_ROI:
            return
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
    # Escoge velocidad de vehículo si existe; si no, usa la peatonal
    speed_val = None
    if track_id in vehicle_speeds:
        speed_val = vehicle_speeds[track_id].get("velocidad")
    elif track_id in track_speed_kmh:
        speed_val = round(track_speed_kmh[track_id], 2)

    vehicle_final_info[track_id] = {
        "tipo": label,
        "roi": final_roi if final_roi else "desconocido",
        "velocidad": speed_val,
        "color": color_live,            # como ya lo tienes
        "color_best": color_best,
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


def crosswalk_id_by_point(pt, polys):
    for i, poly in enumerate(polys, start=1):
        if point_in_polygon(pt, poly):
            return f"XW{i}"
    return None

def log_infraction(timestamp_str, ms_video, track_id, roi, infraccion, valor, umbral, img_path, nota=""):
    with open(INFRA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([timestamp_str, int(ms_video), track_id, roi, infraccion, valor or "", umbral or "", img_path or "", nota or ""])

def _save_infraction_snapshot(frame, x1, y1, x2, y2, prefix):
    h, w = frame.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        pad = 10
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w-1, x2 + pad); y2 = min(h-1, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fname = f"{prefix}.jpg"
    fpath = os.path.join(INFRA_IMG_DIR, fname)
    cv2.imwrite(fpath, crop)
    return fpath

def emit_infraction_once(track_id, infr_code, ms_video, ts_str, roi, frame, bbox_xyxy, valor=None, umbral=None, nota=""):
    key = (track_id, infr_code)
    if key in infra_seen:
        return False
    img_path = _save_infraction_snapshot(frame, *bbox_xyxy, f"{infr_code}_tid{track_id}_{int(ms_video)}")
    log_infraction(ts_str, ms_video, track_id, roi, infr_code, valor, umbral, img_path, nota)
    infra_seen.add(key)
    return True

def emit_infraction_cooldown(track_id, infr_code, ms_video, ts_str, roi, frame, bbox_xyxy, valor=None, umbral=None, nota="", cooldown_ms=INFRA_COOLDOWN_MS):
    key = (track_id, infr_code)
    last = infra_last_ms.get(key, -1e18)
    if ms_video - last < cooldown_ms:
        return False
    img_path = _save_infraction_snapshot(frame, *bbox_xyxy, f"{infr_code}_tid{track_id}_{int(ms_video)}")
    log_infraction(ts_str, ms_video, track_id, roi, infr_code, valor, umbral, img_path, nota)
    infra_last_ms[key] = ms_video
    return True

def _mid(p1, p2):
    """Devuelve el punto medio entre dos puntos (x, y)."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def recompute_ab_unit():
    """Recalcula el vector unitario A->B usando line_start y line_end."""
    global AB_UNIT
    if len(line_start) == 2 and len(line_end) == 2:
        A_mid = _mid(line_start[0], line_start[1])
        B_mid = _mid(line_end[0], line_end[1])
        vx, vy = (B_mid[0]-A_mid[0], B_mid[1]-A_mid[1])
        norm = (vx*vx + vy*vy) ** 0.5
        AB_UNIT = (vx/(norm+1e-9), vy/(norm+1e-9))
    else:
        AB_UNIT = None

def moving_towards_A(prevc, currc, min_proj_px=2.0):
    """Proyección del movimiento sobre el eje A→B. Negativo => hacia A."""
    if AB_UNIT is None or prevc is None:
        return False
    dx = currc[0]-prevc[0]; dy = currc[1]-prevc[1]
    proj = dx*AB_UNIT[0] + dy*AB_UNIT[1]
    return proj < -min_proj_px

# === Helpers de semáforo para infracciones ===
def get_primary_light_snapshot(cfg, semaforo_state):
    pid = (cfg.get("primary_signal_id") or "").upper()
    s = semaforo_state.get_view(pid)
    return {
        "id": pid,
        "state": s.state,              # "RED" | "YELLOW" | "GREEN" | "GRAY"
        "icon": (s.icon or ""),        # "CROS"/"ARRU"/...
        "ts_mqtt": s.ts                # epoch del último update recibido
    }

def light_is_red(snapshot: dict) -> bool:
    # Solo sancionamos con ROJO real (no GRAY); si quieres incluir GRAY como “desconocido”, cámbialo aquí.
    return snapshot and snapshot.get("state") == "RED"


#Carga de modelos YOLOV8
_device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
model_det = YOLO('yolov8m.pt').to(_device) 

#deep_sort = DeepSort(max_age=10, n_init=3, nms_max_overlap=0.5, max_cosine_distance=0.2, nn_budget=None)
deep_sort = DeepSort(
    max_age=30,          # aguanta huecos de hasta ~1s a 30 fps
    n_init=2,            # confirma tracks más rápido
    nms_max_overlap=0.5,
    max_cosine_distance=0.3,  # un pelín más tolerante
    nn_budget=100
)



# Hora de grabación (si no hay metadato, usa ahora)
video_start_time = obtener_hora_grabacion(VIDEO_SOURCE) or datetime.now()

# ==== Identidad de esta ejecución y rutas de salida ====
BASE = os.path.splitext(os.path.basename(VIDEO_SOURCE))[0]
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Infracciones: carpeta por ejecución (separada) ===
INFRA_RUN_DIR = os.path.join("infracciones", f"{BASE}_{RUN_TAG}")
INFRA_IMG_DIR = os.path.join(INFRA_RUN_DIR, "img")
os.makedirs(INFRA_IMG_DIR, exist_ok=True)

INFRA_CSV_PATH = os.path.join(INFRA_RUN_DIR, "infracciones.csv")
# Cabecera de infracciones
with open(INFRA_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","ms_video","track_id","roi","infraccion","valor","umbral","img_path","nota"])
# Intentar cargar geometría desde JSON asociado al video
geo = load_geometry_for_video(VIDEO_SOURCE)

if not geo:
    # Si no hay JSON, intenta abrir tu definidor externo (define.py) para crearlo
    try_run_define(VIDEO_SOURCE)
    # e intenta cargar nuevamente
    geo = load_geometry_for_video(VIDEO_SOURCE)

# 👇 AÑADE ESTO:
if not geo:
    print("[geom] No se pudo obtener geometría (define cancelado o falló). Abortando.")
    raise SystemExit(1)   # evita iniciar hilos sin FULL_W/FULL_H
# 👆 AÑADE ESTO

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
    # Paso peatonal / Líneas / Distancia
    # Acepta el nuevo formato ("paso_peatonal": polígono único)
    # y mantiene compatibilidad con el anterior ("crosswalks": [polígonos])
    paso = geo.get("paso_peatonal")
    if paso:
        # nuevo: un solo polígono -> lo normalizamos a lista de un polígono
        CROSSWALKS_POLYS = [[tuple(map(int, p)) for p in paso]]
    else:
        # legacy: lista de "crosswalks"
        CROSSWALKS_POLYS = [[tuple(map(int, p)) for p in poly] for poly in geo.get("crosswalks", [])]

    # --- ZONA PROHIBIDA DE CAMBIO (opcional) ---
    # Si NO la dibujas en define.py, quedará vacía y la infracción se desactiva.
    ZPC_RAW = geo.get("zona_prohibida_cambio", [])
    ZONA_CC_POLY = [tuple(map(int, p)) for p in ZPC_RAW] if ZPC_RAW else []

    # Líneas A/B y STOP
    A_raw = geo["lines"]["speed_AB"].get("A", [])
    B_raw = geo["lines"]["speed_AB"].get("B", [])
    line_start = [tuple(map(int, A_raw[0])), tuple(map(int, A_raw[1]))] if len(A_raw) == 2 else []
    line_end   = [tuple(map(int, B_raw[0])), tuple(map(int, B_raw[1]))] if len(B_raw) == 2 else []

    STOP_LINE_RAW = geo["lines"].get("stop_line", [])
    STOP_LINE = [tuple(map(int, STOP_LINE_RAW[0])), tuple(map(int, STOP_LINE_RAW[1]))] if len(STOP_LINE_RAW) == 2 else []

    speed_distance_m = float(geo.get("speed_distance_m", speed_distance_m))
    CROSSWALK_DISTANCE_M = float(geo.get("crosswalk_distance_m", 0.0))

    recompute_ab_unit()  # calcula AB_UNIT a partir de line_start/line_end

else:
    # Si no hay JSON válido, evita NameError y sigue sin STOP/PASO/ZPC
    CROSSWALKS_POLYS = []
    STOP_LINE = []
    ZONA_CC_POLY = []   # ← añade esta línea



# =========================
# Semáforo: Config + Estado + SSH
# =========================
cfg = load_signals_config("signals_config.json")

# Persistencia de la luz real y salud de link:
# - hold_state_sec: cuánto dura el último color real antes de pasar a GRAY si no hay nuevas líneas
# - link_stale_sec: latido de conexión (solo para LED, no afecta el color mostrado)
semaforo_state = SemaforoState(
    stale_after_sec=cfg.get("stale_after_sec", 5.0),     # ya no se usa para color; lo mantenemos por compat
    hold_state_sec=cfg.get("hold_state_sec", 30.0),      # <<-- AJUSTABLE
    link_stale_sec=cfg.get("link_stale_sec", 3.0)        # <<-- AJUSTABLE
)
semaforo_state.bootstrap(cfg.get("bootstrap_state", {}))

# Arranque de la lectura SSH (journalctl -f)
signals_ids = [s["id"] for s in cfg["signals"]]
ssh_cfg = cfg["ssh"]
ssh_thread = SSHSemaforoBridge(
    host=ssh_cfg["host"],
    username=ssh_cfg["username"],
    password=ssh_cfg["password"],
    command=ssh_cfg.get("command", "journalctl -u scheduler-mqtt -f"),
    topic_prefix=ssh_cfg.get("parse", {}).get("topic_prefix", "esp32/semaforos/"),
    signals_ids=signals_ids,
    state=semaforo_state,
    port=ssh_cfg.get("port", 22)
)
ssh_thread.start()
# =========================

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
    global rois, FULL_H, FULL_W, output_dir, previous_centers

    while not stop_event.is_set() or not raw_queue.empty():
        try:
            frame, pos_msec = raw_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        live_counts = Counter()
        live_now = set()  # <- tracks presentes en este frame

        # Detección en una sola pasada
        det = run_yolo_scaled(frame, model_det, DETECTION_DOWNSCALE)

        detections = []
        for (x1, y1, x2, y2, confidence, cls_name) in det:
            # Filtra solo clases de interés (ajusta a tu caso)
            if cls_name not in {"car", "bus", "truck", "motorbike", "bicycle", "person"}:
                continue

            frame_h, frame_w = frame.shape[:2]
            box_w = x2 - x1; box_h = y2 - y1
            min_ratio = 0.015 if cls_name == "person" else 0.02
            if box_w < frame_w * min_ratio or box_h < frame_h * min_ratio:
                continue
            if box_w > frame_w * 0.8 or box_h > frame_h * 0.8:
                continue
            if confidence < 0.4:
                continue

            # ⚠️ Primero calcula el centro de la detección (NO del track)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # FILTRO: solo aceptamos detecciones cuyo centro caiga dentro de algún ROI
            roi_id = roi_by_point(cx, cy, rois, FULL_W, FULL_H)
            if roi_id is None:
                continue

            # Etiqueta para el tracker (si quieres mantener el sufijo ROI, déjalo igual)
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
                    # snapshot del semáforo propio en el instante del cruce
                    light_snap = get_primary_light_snapshot(cfg, semaforo_state)
                    if light_is_red(light_snap):
                        # 🚨 Infracción: cruzó línea de pare en ROJO
                        if emit_infraction_cooldown(
                            track_id, "STOPLINE_RED", pos_msec, timestamp_str,
                            roi_for_event, frame, bbox_xyxy,
                            valor="", umbral="", nota="Cruzó línea de pare con luz ROJA"
                        ):
                            cv2.putText(frame, "STOPLINE_RED", (x1, max(20, y1-25)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # 2) Bloqueo (detenido sobre la línea)
                dist_px = abs(side_of_line(currc, A, B)) / ( ((B[0]-A[0])**2 + (B[1]-A[1])**2)**0.5 + 1e-9 )
                spd_px  = center_speed_px(currc, prevc)
                if dist_px <= 3.0 and spd_px <= 1.5:
                    cnt = on_stop_frames.get(track_id, 0) + 1
                    on_stop_frames[track_id] = cnt
                    if cnt == STOP_MIN_FRAMES:
                        emit_infraction_cooldown(
                            track_id, "STOPLINE_BLOCK", pos_msec, timestamp_str,
                            roi_for_event, frame, bbox_xyxy,
                            valor="", umbral=f"≥{STOP_MIN_FRAMES} frames",
                            nota="Detenido sobre la línea de pare"
                        )
                        cv2.putText(frame, "STOPLINE_BLOCK", (x1, max(20, y1-25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                else:
                    on_stop_frames[track_id] = 0

            # --- CROSSWALK (bloqueo por detención en paso peatonal) ---
            en_cebra = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS)
            prevc = track_prev_center.get(track_id)
            if en_cebra and center_speed_px((cx, cy), prevc) <= CROSSWALK_STOP_PX:
                cnt = on_crosswalk_frames.get(track_id, 0) + 1
                on_crosswalk_frames[track_id] = cnt
                if cnt == CROSSWALK_MIN_FRAMES:
                    emit_infraction_cooldown(
                        track_id, "CROSSWALK_BLOCK", pos_msec, timestamp_str,
                        roi_for_event, frame, bbox_xyxy,
                        valor="", umbral=f"≥{CROSSWALK_MIN_FRAMES} frames",
                        nota="Detenido sobre paso peatonal"
                    )
                    cv2.putText(frame, "CROSSWALK_BLOCK", (x1, max(20, y1-25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                on_crosswalk_frames[track_id] = 0

            # ===== CRUCE (entrada) de PASO PEATONAL =====
            en_cebra = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS)
            was_in = prev_in_crosswalk.get(track_id, False)

            # ROI para el CSV (puede ser ROI del carril, o el paso peatonal detectada)
            roi_for_event = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2) or "fuera"
            xw_id = crosswalk_id_by_point((cx, cy), CROSSWALKS_POLYS) or ""

            bbox_xyxy = (x1, y1, x2, y2)

            # 3.1 Entrada a paso peatonal (solo transición False -> True)
            if en_cebra and not was_in:
                # identifica paso peatonal y snapshot de luz
                xw_id = crosswalk_id_by_point((cx, cy), CROSSWALKS_POLYS) or ""
                light_snap = get_primary_light_snapshot(cfg, semaforo_state)

                if light_is_red(light_snap):
                    # 🚨 Infracción: entró al paso peatonal en ROJO
                    note = "Entró a paso peatonal en ROJO" + (f" ({xw_id})" if xw_id else "")
                    if emit_infraction_cooldown(
                        track_id, "CROSSWALK_RED", pos_msec, timestamp_str,
                        roi_for_event, frame, bbox_xyxy,
                        valor="", umbral="", nota=note
                    ):
                        cv2.putText(frame, "CROSSWALK_RED", (x1, max(20, y1-25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # Actualiza estado de paso peatonal para el próximo frame
            prev_in_crosswalk[track_id] = en_cebra

            
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

            # ========= 3.3 PEATÓN: cruce fuera del paso (PEDESTRIAN_OUTSIDE_CROSSWALK) =========
            # Requiere que ya existan: rois, CROSSWALKS_POLYS, pos_msec, timestamp_str, roi_for_event, (x1,y1,x2,y2), cx, cy
            is_person = (label == "person")
            if is_person:
                # ¿El centro del peatón está dentro de la calzada (cualquier ROI)?
                in_road = any(point_in_polygon((cx, cy), r["poly"]) for r in rois)
                # ¿Está dentro del paso peatonal (si existe)?
                in_paso = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS) if CROSSWALKS_POLYS else False

                was_in_road = prev_in_road_person.get(track_id, False)

                # 1) Entrada a calzada SIN paso → inicia sesión
                if in_road and not was_in_road and not in_paso:
                    ped_session[track_id] = {
                        "start_pos": (cx, cy),
                        "start_ms": pos_msec,
                        "frames": 0,
                        "ever_in_paso": False
                    }

                # 2) Actualización de sesión, y evaluación al salir de la calzada
                if track_id in ped_session:
                    sess = ped_session[track_id]
                    sess["frames"] += 1
                    sess["ever_in_paso"] = sess["ever_in_paso"] or in_paso

                    # Si ya NO está en calzada, evaluar si fue cruce fuera del paso
                    if not in_road:
                        dx = cx - sess["start_pos"][0]; dy = cy - sess["start_pos"][1]
                        dist_px = (dx*dx + dy*dy) ** 0.5
                        if (sess["frames"] >= PEDESTRIAN_CROSS_MIN_FRAMES
                            and dist_px >= PEDESTRIAN_CROSS_MIN_PX
                            and not sess["ever_in_paso"]):
                            note = "Peatón cruzó fuera del paso peatonal"
                            emit_infraction_once(
                                track_id, "PEDESTRIAN_OUTSIDE_CROSSWALK",
                                pos_msec, timestamp_str,
                                roi_for_event, frame, (x1, y1, x2, y2),
                                valor="", umbral="", nota=note
                            )
                            cv2.putText(frame, "PEDESTRIAN_OUTSIDE_CROSSWALK",
                                        (x1, max(20, y1-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        # Cerrar sesión al salir
                        ped_session.pop(track_id, None)

                # Persistir estado previo
                prev_in_road_person[track_id] = in_road
            # ========= /PEATÓN: cruce fuera del paso =========

            # ===== Velocidad PEATÓN (distancia del paso / tiempo) =====
            # Requisitos: leer CROSSWALK_DISTANCE_M del JSON y tener CROSSWALKS_POLYS.
            if label == "person" and CROSSWALKS_POLYS and CROSSWALK_DISTANCE_M > 0:
                in_paso = any(point_in_polygon((cx, cy), poly) for poly in CROSSWALKS_POLYS)
                was_in_paso = prev_in_paso_person.get(track_id, False)

                # Entrada al paso: arrancar cronómetro
                if in_paso and not was_in_paso:
                    ped_in_paso_ms[track_id] = pos_msec

                # Salida del paso: calcular velocidad y guardarla para el CSV
                if (not in_paso) and was_in_paso:
                    start_ms = ped_in_paso_ms.pop(track_id, None)
                    if start_ms is not None:
                        dt_s = max(1e-3, (pos_msec - start_ms) / 1000.0)
                        v_kmh = (CROSSWALK_DISTANCE_M / dt_s) * 3.6
                        cv2.putText(frame, f"v_ped={v_kmh:.1f} km/h",
                            (x1, max(20, y1-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                        track_speed_kmh[track_id] = v_kmh  # se usará en la misma columna 'velocidad' del CSV
                        vehicle_speeds[track_id] = {
                            "velocidad": round(v_kmh, 2),
                            "timestamp": timestamp_str,
                            "color": vehicle_colors.get(track_id, "desconocido")
                        }
                prev_in_paso_person[track_id] = in_paso
            # ===== /Velocidad PEATÓN =====




            track_id_to_label[track_id] = label
            live_now.add(track_id)
            last_seen_msec[track_id] = pos_msec

            # --- transición de estado ROI ---
            prev_state = vehicle_roi_state.get(track_id, "fuera")
            vehicle_roi_state[track_id] = actual_roi  # "ROIx" o "fuera"

            # --- CAMBIO DE CANAL PROHIBIDO ---
            if prev_state.startswith("ROI") and actual_roi.startswith("ROI") and prev_state != actual_roi:
                # Debe ocurrir DENTRO de la ZPC (si está definida)
                en_zona_cc = False
                if ZONA_CC_POLY:
                    en_zona_cc = point_in_polygon((cx, cy), ZONA_CC_POLY)
                # Si NO hay ZPC definida, NO dispares esta infracción (normativa del prof. Nevado)
                if en_zona_cc:
                    note = f"Cambio de canal prohibido: {prev_state} → {actual_roi}"
                    if emit_infraction_once(track_id, "LANE_CHANGE_FORBIDDEN", pos_msec, timestamp_str,
                                            actual_roi, frame, bbox_xyxy,
                                            valor="", umbral="", nota=note):
                        cv2.putText(frame, "LANE_CHANGE_FORBIDDEN", (x1, max(20, y1-50)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

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

                    # Confirmar contravía B->A dentro de ventana
                    prog = wrongway_prog.get(track_id)
                    if prog and prog.get("first") == "END" and (pos_msec - prog["t_first"]) <= WRONGWAY_MAX_MS:
                        roi_for_event = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2) or "fuera"
                        bbox_xyxy = (x1, y1, x2, y2)
                        note = "Ingresó por B y retrocedió hacia A"
                        emit_infraction_once(track_id, "WRONG_WAY", pos_msec, timestamp_str,
                                            roi_for_event, frame, bbox_xyxy, valor="", umbral="", nota=note)
                        cv2.putText(frame, "WRONG_WAY", (x1, max(20, y1-55)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # limpiar en cualquier caso
                    wrongway_prog.pop(track_id, None)


            if len(line_end) == 2:
                # Centros para chequear cruce y dirección
                prevc = track_prev_center.get(track_id)
                currc = (cx, cy)

                # 1) Detectar cruce de B SIEMPRE (independiente de vehicle_times)
                if prevc is not None and cruzo_linea(line_end[0], line_end[1], prevc, currc):
                    # Debug opcional de proyección
                    if AB_UNIT is not None:
                        dx, dy = currc[0]-prevc[0], currc[1]-prevc[1]
                        proj = dx*AB_UNIT[0] + dy*AB_UNIT[1]
                        print(f"[DBG] B-cross tid={track_id} proj={proj:.2f} (negativo=hacia A) ms={int(pos_msec)}")

                    # Si aún NO hay 'start' (no cruzó A), y se mueve hacia A => siembra B-primero
                    vt = vehicle_times.get(track_id)
                    start_exists = (vt is not None) and (vt.get('start') is not None)
                    if not start_exists and moving_towards_A(prevc, currc):
                        wrongway_prog.setdefault(track_id, {"first":"END", "t_first": pos_msec})

                    # 2) Sólo si YA HAY 'start' y NO tiene 'end', usamos B como fin para velocidad
                    if start_exists and vt.get('end') is None:
                        vehicle_times[track_id]['end'] = pos_msec

                        # arma timestamp aquí (mismo scope que speed)
                        event_time = video_start_time + timedelta(milliseconds=pos_msec)
                        timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')

                        t1 = vehicle_times[track_id]['start']
                        t2 = vehicle_times[track_id]['end']
                        if t1 is not None:
                            elapsed = (t2 - t1) / 1000.0
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
                                cv2.putText(frame, f"END: {track_id}", (10, 140),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                                # 🚨 Exceso de velocidad — MOVER AQUÍ ADENTRO
                                roi_for_event = roi_by_bbox(x1, y1, x2, y2, rois, min_ratio=0.2) or "fuera"
                                bbox_xyxy = (x1, y1, x2, y2)
                                if speed > SPEED_LIMIT_KMH:
                                    note = f"Vel {speed:.2f} km/h > {SPEED_LIMIT_KMH:.2f}"
                                    if emit_infraction_once(track_id, "SPEEDING", pos_msec, timestamp_str,
                                                            roi_for_event, frame, bbox_xyxy,
                                                            valor=f"{speed:.2f}", umbral=f"{SPEED_LIMIT_KMH:.2f}", nota=note):
                                        cv2.putText(frame, f"SPEEDING {speed:.1f} km/h", (x1, y1-30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

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

            # 3.3 SIEMPRE actualiza el centro al final
            track_prev_center[track_id] = (cx, cy)

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
                    wrongway_prog.pop(tid, None)
                    ped_session.pop(tid, None)
                    prev_in_road_person.pop(tid, None)
                    ped_speed.pop(tid, None)
                    prev_in_paso_person.pop(tid, None)
                    track_speed_kmh.pop(tid, None)


                    

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

        # STOP-LINE (ROJO)
        if STOP_LINE and len(STOP_LINE) == 2:
            cv2.line(frame, STOP_LINE[0], STOP_LINE[1], (0, 0, 255), 2)
            cv2.putText(frame, "STOP", STOP_LINE[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        # PASO PEATONAL (magenta)
        for poly in CROSSWALKS_POLYS:
            if len(poly) >= 3:
                arr = np.array(poly, np.int32)
                cv2.polylines(frame, [arr], True, (255, 0, 255), 2)
        
        # ZCP Zona de cambio prohibido
        if ZONA_CC_POLY:
            cv2.polylines(frame, [np.array(ZONA_CC_POLY, dtype=np.int32)], True, (255,0,255), 2)
            cxz, cyz = np.mean(np.array(ZONA_CC_POLY), axis=0).astype(int)
            cv2.putText(frame, "ZONA CC", (int(cxz), int(cyz)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)


        # === Overlay de semáforo (persistente por hold_state_sec, LED de link) ===
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
try:
    ssh_thread.stop()
except Exception:
    pass

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
