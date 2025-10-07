import cv2, os, json, argparse, numpy as np
from datetime import datetime
import shutil

# =================== util ===================
def ensure_dirs():
    os.makedirs("config_geom", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)

def draw_text_lines(img, lines, x=10, y=22, dy=22, color=(255,255,255)):
    for ln in lines:
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += dy

def draw_numbered_points(vis, pts, color):
    for i, p in enumerate(pts, 1):
        cv2.circle(vis, p, 5, (0,0,0), -1)
        cv2.circle(vis, p, 7, color, 2)
        cv2.putText(vis, str(i), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_all(frame, state):
    vis = frame.copy()

    # ROIs (amarillo)
    for idx, poly in enumerate(state["rois"], 1):
        arr = np.array(poly, np.int32)
        cv2.polylines(vis, [arr], True, (0,255,255), 3)
        draw_numbered_points(vis, poly, (0,255,255))
        c = np.mean(arr, axis=0).astype(int)
        cv2.putText(vis, f"ROI{idx}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    # PASO PEATONAL (cyan)
    if state.get("paso_peatonal"):
        arr = np.array(state["paso_peatonal"], np.int32)
        cv2.polylines(vis, [arr], True, (255,255,0), 3)
        draw_numbered_points(vis, state["paso_peatonal"], (255,255,0))
        c = np.mean(arr, axis=0).astype(int)
        cv2.putText(vis, "PASO", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)

    # ZONA CAMBIO PROHIBIDO (magenta)
    if state.get("zona_cambio_prohibido"):
        arr = np.array(state["zona_cambio_prohibido"], np.int32)
        cv2.polylines(vis, [arr], True, (255,0,255), 2)
        draw_numbered_points(vis, state["zona_cambio_prohibido"], (255,0,255))
        c = np.mean(arr, axis=0).astype(int)
        cv2.putText(vis, "ZONA CC", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2, cv2.LINE_AA)


    # Línea A y B (naranja)
    if len(state["line_A"]) == 2:
        cv2.line(vis, state["line_A"][0], state["line_A"][1], (0,200,255), 3)
        draw_numbered_points(vis, state["line_A"], (0,200,255))
        cv2.putText(vis, "A", state["line_A"][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)
    if len(state["line_B"]) == 2:
        cv2.line(vis, state["line_B"][0], state["line_B"][1], (0,200,255), 3)
        draw_numbered_points(vis, state["line_B"], (0,200,255))
        cv2.putText(vis, "B", state["line_B"][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

    # Stop-line (verde)
    if len(state["stop_line"]) == 2:
        cv2.line(vis, state["stop_line"][0], state["stop_line"][1], (0,200,0), 3)
        draw_numbered_points(vis, state["stop_line"], (0,200,0))
        cv2.putText(vis, "STOP", state["stop_line"][0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2, cv2.LINE_AA)

    # En edición (resaltado)
    if state["mode"] in ("ROI", "PASO", "ZPC") and len(state["curr_poly"]) >= 1:
        cv2.polylines(vis, [np.array(state["curr_poly"], np.int32)], False, (0,0,255), 3)
        draw_numbered_points(vis, state["curr_poly"], (0,0,255))

    if state["mode"] in ("LINE_A","LINE_B","STOP") and len(state["curr_line"]) >= 1:
        draw_numbered_points(vis, state["curr_line"], (255,0,255))
        if len(state["curr_line"]) == 2:
            cv2.line(vis, state["curr_line"][0], state["curr_line"][1], (255,0,255), 3)

    # HUD

    pasos = 1 if state.get("paso_peatonal") else 0
    zpc = 1 if state.get("zona_cambio_prohibido") else 0
    mode_name = {"ROI":"ROI","PASO":"PASO","LINE_A":"LINEA A","LINE_B":"LINEA B","STOP":"STOP-LINE","ZPC": "ZONA CC","DONE": "FINAL"}[state["mode"]]
    hud = [
        f"Modo: {mode_name}   |   ROIs {len(state['rois'])}/{state['n_rois_target']}   | Paso {pasos} | ZPC {zpc}",
        f"A: {'OK' if len(state['line_A'])==2 else '—'}   B: {'OK' if len(state['line_B'])==2 else '—'}   STOP: {'OK' if len(state['stop_line'])==2 else '—'}",
        "Controles: Click=agregar punto  |  Enter=confirmar  |  Backspace=deshacer  |  R=reiniciar",
        "           M=siguiente paso      |  S=guardar y salir  |  ESC=salir sin guardar",
    ]
    draw_text_lines(vis, hud, x=10, y=22)
    return vis

def confirm_current(state):
    """Cierra el elemento en edición si es válido."""
    if state["mode"] in ("ROI","PASO","ZPC"):
        if len(state["curr_poly"]) >= 3:
            if state["mode"] == "ROI":
                state["rois"].append(state["curr_poly"].copy())
            elif state["mode"] == "PASO":
                state["paso_peatonal"] = state["curr_poly"].copy()
            else:  # ZPC
                state["zona_cambio_prohibido"] = state["curr_poly"].copy()
            state["curr_poly"].clear()
    elif state["mode"] in ("LINE_A","LINE_B","STOP"):
        if len(state["curr_line"]) == 2:
            if state["mode"] == "LINE_A":
                state["line_A"] = state["curr_line"].copy()
            elif state["mode"] == "LINE_B":
                state["line_B"] = state["curr_line"].copy()
            else:
                state["stop_line"] = state["curr_line"].copy()
            state["curr_line"].clear()

def advance_mode(state):
    """Avanza ROI -> PASO -> LINE_A -> LINE_B -> STOP (valida mínimos por paso)."""
    if state["mode"] == "ROI":
        if len(state["rois"]) < state["n_rois_target"]:
            print(f"⚠ Faltan ROIs ({len(state['rois'])}/{state['n_rois_target']}). Confirma con Enter cada ROI.")
            return
        state["mode"] = "PASO"
        state["curr_poly"].clear(); state["curr_line"].clear()
        print("→ Modo PASO PEATONAL (un solo polígono; Enter confirma; M para seguir).")
    elif state["mode"] == "PASO":
        state["mode"] = "LINE_A"
        state["curr_poly"].clear(); state["curr_line"].clear()
        print("→ Modo LINEA A (2 clics + Enter).")
    elif state["mode"] == "LINE_A":
        if len(state["line_A"]) != 2:
            print("⚠ Falta confirmar LINEA A (Enter).")
            return
        state["mode"] = "LINE_B"
        state["curr_line"].clear()
        print("→ Modo LINEA B (2 clics + Enter).")
    elif state["mode"] == "LINE_B":
        if len(state["line_B"]) != 2:
            print("⚠ Falta confirmar LINEA B (Enter).")
            return
        state["mode"] = "STOP"
        state["curr_line"].clear()
        print("→ Modo STOP-LINE (2 clics + Enter). Si no quieres, presiona M para saltar.")
    elif state["mode"] == "STOP":
        if len(state["stop_line"]) != 2:
            print("⚠ Falta confirmar STOP (Enter).")
            return
        # NUEVO: ir a ZPC
        state["mode"] = "ZPC"
        state["curr_poly"].clear(); state["curr_line"].clear()
        print("→ Modo ZPC (Zona Prohibida de Cambio): dibuja 0..1 polígono; Enter confirma; M para terminar).")
    elif state["mode"] == "ZPC":
        # Terminar y pasar a guardar
        state["mode"] = "DONE"
        print("→ Guardando geometría...")
        print("↪ Ya estás en el último paso. Presiona S para guardar o ESC para salir sin guardar.")

def save_json_and_preview(video_path, frame0, state, dist_ab):
    ensure_dirs()
    data = {
        "video_source": video_path,
        "frame_size": [int(frame0.shape[0]), int(frame0.shape[1])],  # H, W
        "speed_distance_m": float(dist_ab),
        "rois": state["rois"],
        # Nuevo campo único:
        "paso_peatonal": state["paso_peatonal"],
        # Compatibilidad con versiones anteriores:
        "crosswalks": ([state["paso_peatonal"]] if state["paso_peatonal"] else []),
        "zona_prohibida_cambio": state["zona_cambio_prohibido"],     # ← NUEVO
        "lines": {
            "speed_AB": {"A": state["line_A"], "B": state["line_B"]},
            "stop_line": state["stop_line"],
        },
        "timestamp": datetime.now().isoformat(),
        "crosswalk_distance_m": float(state.get("crosswalk_distance_m", 0.0)),
    }
    with open("config_geom/geometry_all.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # preview limpia (sin HUD)
    vis = frame0.copy()
    # ROIs
    for poly in state["rois"]:
        arr = np.array(poly, np.int32)
        cv2.polylines(vis, [arr], True, (0,255,255), 2)
    # PASO PEATONAL
    if state["paso_peatonal"]:
        arr = np.array(state["paso_peatonal"], np.int32)
        cv2.polylines(vis, [arr], True, (255,255,0), 2)
    # Líneas
    if len(state["line_A"]) == 2:
        cv2.line(vis, state["line_A"][0], state["line_A"][1], (0,200,255), 2)
    if len(state["line_B"]) == 2:
        cv2.line(vis, state["line_B"][0], state["line_B"][1], (0,200,255), 2)
    if len(state["stop_line"]) == 2:
        cv2.line(vis, state["stop_line"][0], state["stop_line"][1], (0,200,0), 2)

    cv2.imwrite("resultados/geom_preview.png", vis)
    print("✔ Guardado: config_geom/geometry_all.json")
    print("✔ Preview : resultados/geom_preview.png")

# ================== main ==================
def main():
    ap = argparse.ArgumentParser(description="Definir ROIs, Paso Peatonal, Líneas A/B y Stop en UNA ventana; flujo por consola.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--nrois", type=int, default=None)
    args = ap.parse_args()

    # 1) Solo nrois por consola (antes de abrir ventana)
    if args.nrois is None:
        while True:
            try:
                args.nrois = int(input("¿Cuántos ROIs quieres definir? ").strip())
                if args.nrois <= 0: raise ValueError
                break
            except:
                print("Ingresa un entero positivo.")

    # 2) Abrir primer frame y ventana única
    cap = cv2.VideoCapture(args.video)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("No pude leer el primer frame del video.")

    win = "Definir Geometría"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # ventana redimensionable
    cv2.resizeWindow(win, 1280, 720)          # tamaño inicial cómodo (no reescala la imagen)

    # 3) Estado
    state = {
        "mode": "ROI",                   # ROI -> PASO -> LINE_A -> LINE_B -> STOP
        "n_rois_target": args.nrois,
        "rois": [],
        "paso_peatonal": [],             # ← antes era "crosswalks": []
        "line_A": [],
        "line_B": [],
        "stop_line": [],
        "zona_cambio_prohibido": [],     # ← NUEVO (un solo polígono, opcional)"zona_cambio_prohibido": []
        "curr_poly": [],
        "curr_line": [],
    }

    # 4) Mouse (agregar puntos)
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["mode"] in ("ROI","PASO","ZPC"):
                state["curr_poly"].append((int(x), int(y)))
            elif state["mode"] in ("LINE_A","LINE_B","STOP"):
                if len(state["curr_line"]) < 2:
                    state["curr_line"].append((int(x), int(y)))

    cv2.setMouseCallback(win, on_mouse)

    # 5) Loop de interfaz (sin input())
    print("→ Modo ROI: dibuja cada ROI (polígono). Enter: confirmar uno. M: pasar a Paso Peatonal cuando completes los requeridos.")
    while True:
        if state["mode"] == "DONE":
            break
        vis = draw_all(frame0, state)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF

        if k in (13,10):           # Enter
            confirm_current(state)
        elif k == ord('m'):        # siguiente paso
            advance_mode(state)
        elif k == 8:               # Backspace
            if state["mode"] in ("ROI","PASO","ZPC") and state["curr_poly"]:
                state["curr_poly"].pop()
            elif state["mode"] in ("LINE_A","LINE_B","STOP") and state["curr_line"]:
                state["curr_line"].pop()
        elif k == ord('r'):        # reiniciar elemento en edición
            state["curr_poly"].clear()
            state["curr_line"].clear()
        elif k == ord('s'):        # guardar y salir
            if len(state["rois"]) < state["n_rois_target"]:
                print(f"⚠ Debes definir al menos {state['n_rois_target']} ROIs.")
                continue
            if len(state["line_A"]) != 2 or len(state["line_B"]) != 2:
                print("⚠ Debes definir y confirmar LINEA A y LINEA B (Enter).")
                continue
            break
        elif k == 27:              # ESC: salir sin guardar
            print("✖ Saliste sin guardar.")
            cv2.destroyWindow(win)
            return

    cv2.destroyWindow(win)

    # 6) AHORA pedir distancia A–B por consola (ventana ya cerrada)
    while True:
        try:
            dist_ab = float(input("Distancia real entre A y B (metros): ").strip())
            if dist_ab <= 0: raise ValueError
            break
        except:
            print("Ingresa un número positivo (ej. 10.0).")


    # 6.1) NUEVO: distancia del paso peatonal (opcional)
    crosswalk_distance_m = 0.0
    if state.get("paso_peatonal"):  # solo tiene sentido si dibujaste el paso
        while True:
            s = input("Distancia del paso peatonal en metros (Enter para omitir): ").strip()
            if s == "":
                break
            try:
                v = float(s)
                if v <= 0:
                    print("Debe ser un número positivo (ej. 6.5).")
                    continue
                crosswalk_distance_m = v
                break
            except:
                print("Ingresa un número válido (ej. 6.5).")

    # Guarda el valor en el estado para que lo tome save_json_and_preview
    state["crosswalk_distance_m"] = crosswalk_distance_m

    # 7) Guardar JSON + preview
    save_json_and_preview(args.video, frame0, state, dist_ab)
    print("✔ Geometría lista.")


if __name__ == "__main__":
    main()
