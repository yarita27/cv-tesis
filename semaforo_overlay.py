import re
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque


# SSH
import socket
import paramiko

# UI y JSON
import json
import cv2
import numpy as np

# =========================
# Parseo de payload/colores
# =========================
PAYLOAD_RE = re.compile(r"^\s*(RED|YELLOW|GREEN)\s+ICON:([A-Z]+)\s*$", re.IGNORECASE)
#JOURNAL_RE = re.compile(r"\[MQTT\]\s+([^\s]+)\s*=>\s*(.+)$")  # [MQTT] esp32/semaforos/S1 => GREEN ICON:ARRU
JOURNAL_RE = re.compile(r"(esp32/semaforos/[A-Za-z0-9_\-]+).*?(?:=>|:)?\s*(.+)$", re.IGNORECASE)

COLOR_BGR = {
    "RED":    (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN":  (0, 255, 0),
    "GRAY":   (128, 128, 128),
}

# =========================
# Estado por semáforo (ID)
# =========================
# ... imports arriba iguales ...

@dataclass
class SignalSample:
    state: str = "GRAY"
    icon: Optional[str] = None
    ts: float = field(default_factory=time.time)

class SemaforoState:
    """
    Buffer por-ID (S1, S2, ...). Mantiene:
    - último estado e instante por cada ID
    - latido de conexión (last_link_ok_ts)
    """
    def __init__(self, stale_after_sec: float = 5.0, hold_state_sec: float = 30.0, link_stale_sec: float = 3.0):
        self._lock = threading.Lock()
        self._data: Dict[str, SignalSample] = {}
        # (stale_after_sec ya no se usa para grisar la luz)
        self._state_hold = float(hold_state_sec)
        self._link_stale = float(link_stale_sec)
        self._last_link_ok_ts = 0.0
        self.history = {}  # { "S1": deque([(t_srv, "RED"), ...], maxlen=5000) }

    def _store(self, sid, state, t_epoch, icon=None):
        sid = sid.upper()
        # actualiza el estado “en vivo” que usa el HUD
        with self._lock:
            self._data[sid] = SignalSample(state=state, icon=icon, ts=float(t_epoch))
        # además guarda histórico para consultas “al tiempo T”
        dq = self.history.setdefault(sid, deque(maxlen=5000))
        dq.append((float(t_epoch), state))


    def update_from_log_with_ts(self, sid, state, icon, t_override=None):
        # t_override: epoch del SERVIDOR si viene en el log ('ts='). Si no, usa reloj local.
        t = float(t_override) if (t_override is not None) else time.time()
        self._store(sid, state, t, icon=icon)
        # mantén aquí cualquier lógica que ya tengas para heartbeat/enlace

    def get_state_at(self, sid, t_query):
        """Último estado conocido en o antes de t_query (epoch seg, reloj del servidor)."""
        sid = sid.upper()
        dq = self.history.get(sid)
        if not dq:
            return None
        for t, st in reversed(dq):
            if t <= t_query:
                return {"state": st, "t": t}
        return None
    # --- Conectividad ---
    def link_heartbeat(self):
        with self._lock:
            self._last_link_ok_ts = time.time()

    def link_down(self):
        # no borramos estados; solo marcamos que no hay latido
        with self._lock:
            # ponerlo "viejo" para que caiga en stale rápido, pero sin tocar colores
            self._last_link_ok_ts = 0.0

    def link_is_stale(self) -> bool:
        with self._lock:
            if self._last_link_ok_ts <= 0.0:
                return True
            return (time.time() - self._last_link_ok_ts) > self._link_stale

    # --- Estados de luces ---
    def bootstrap(self, snapshot: Dict[str, Dict[str, str]]):
        with self._lock:
            now = time.time()
            for sid, info in snapshot.items():
                st = (info.get("state") or "GRAY").upper()
                ic = info.get("icon")
                self._data[sid.upper()] = SignalSample(state=st, icon=ic, ts=now)

    def update_from_log(self, sid: str, payload: str):
        m = PAYLOAD_RE.match(payload or "")
        if not m:
            return
        state = m.group(1).upper()
        icon = m.group(2).upper()
        with self._lock:
            self._data[sid.upper()] = SignalSample(state=state, icon=icon, ts=time.time())

    def get_view(self, sid: str) -> SignalSample:
        """
        Devuelve el último estado. NO lo vuelve GRAY por conexión.
        Solo lo pone GRAY si el estado está más viejo que hold_state_sec.
        """
        with self._lock:
            s = self._data.get(sid.upper(), SignalSample())
            age = time.time() - s.ts if s.ts else 1e9
            if age > self._state_hold:
                return SignalSample(state="GRAY", icon=s.icon, ts=s.ts)
            return s

# =========================
# Hilo SSH: tail de journal
# =========================
class SSHSemaforoBridge(threading.Thread):
    """
    Conecta por SSH a la Raspberry y sigue el log:
      journalctl -u scheduler-mqtt -f
    Parsea líneas tipo:
      [MQTT] esp32/semaforos/S1 => GREEN ICON:ARRU
    y actualiza el estado del ID correspondiente.
    """
    def __init__(self, host: str, username: str, password: str, command: str,
                 topic_prefix: str, signals_ids, state: SemaforoState,
                 port: int = 22, reconnect_sec: float = 3.0):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.command = command
        self.topic_prefix = topic_prefix.rstrip('/') + '/'
        self.signals_whitelist = set([sid.upper() for sid in signals_ids])  # IDs permitidos (S1, S2, ...)
        self.state = state
        self.reconnect_sec = reconnect_sec
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _handle_line(self, line: str):
        # Ejemplos que recibimos:
        # [MQTT] semaforos/S1 => GREEN ICON:ARRU
        # [MQTT] semaforos/S2 => RED ICON:CROS

        if not line:
            return

        # 1) Localiza el topic usando el prefijo configurado
        idx = line.find(self.topic_prefix)
        if idx < 0:
            return  # no es para nosotros

        # topic va hasta el siguiente espacio (o fin de línea)
        j = line.find(' ', idx)
        if j < 0:
            j = len(line)
        topic = line[idx:j].strip()            # ej: "semaforos/S1"
        sid   = topic.split('/')[-1].upper()   # "S1"

        if self.signals_whitelist and sid not in self.signals_whitelist:
            return

        # 2) Extrae el payload (después de "=>", si existe; si no, lo que quede)
        rest = ""
        if "=>" in line:
            rest = line.split("=>", 1)[1].strip()
        else:
            rest = line[j:].strip()

        # Normalizamos algunas variantes típicas
        rest_up = rest.upper()

        # Caso feliz: "GREEN ICON:ARRU" (o RED/YELLOW)
        if rest_up.startswith(("RED","GREEN","YELLOW")):
            if "ICON:" not in rest_up:
                rest_up = f"{rest_up} ICON:NA"

            # 1) Extrae ts=... si vino en la línea (usa 'rest', no pierdas nada)
            t_server = None
            m_ts = re.search(r"\bts=([0-9]+\.[0-9]+|\d+)\b", rest, flags=re.IGNORECASE)
            if m_ts:
                try:
                    t_server = float(m_ts.group(1))
                except:
                    t_server = None

            # 2) Obtén state y icon del payload
            st = rest_up.split()[0]  # RED/GREEN/YELLOW
            m_icon = re.search(r"ICON:([A-Z0-9_]+)", rest_up)
            ic = m_icon.group(1) if m_icon else "NA"

            # 3) Guarda usando el tiempo del servidor si está
            self.state.update_from_log_with_ts(sid, st, ic, t_override=t_server)
            self.state.link_heartbeat()
            return


        # Variantes "state=GREEN icon=ARRU", JSON, etc. (fallbacks)
        # state=...
        if "STATE=" in rest_up:
            st = None; ic = None
            for part in re.split(r"[,\s]+", rest_up):
                if part.startswith("STATE="):
                    st = part.split("=",1)[1]
                elif part.startswith("ICON=") or part.startswith("PHASE=") or part.startswith("MODE="):
                    ic = part.split("=",1)[1]
            if st in ("RED","GREEN","YELLOW"):
                t_server = None
                m_ts = re.search(r"\bts=([0-9]+\.[0-9]+|\d+)\b", rest, flags=re.IGNORECASE)
                if m_ts:
                    try:
                        t_server = float(m_ts.group(1))
                    except:
                        t_server = None

                self.state.update_from_log_with_ts(sid, st, (ic or 'NA'), t_override=t_server)
                self.state.link_heartbeat()
                return


        # JSON {"state":"GREEN","icon":"ARRU"}
        try:
            obj = json.loads(rest)
            st  = (obj.get("state") or "").upper()
            ic  = (obj.get("icon")  or obj.get("phase") or obj.get("mode") or "NA").upper()
            t_server = obj.get("ts", None)  # si el publicador ya manda "ts" en JSON

            if st in ("RED","GREEN","YELLOW"):
                # intenta convertir ts si vino como string
                try:
                    t_server = float(t_server) if t_server is not None else None
                except:
                    t_server = None

                self.state.update_from_log_with_ts(sid, st, ic, t_override=t_server)
                self.state.link_heartbeat()
                return

        except Exception:
            pass

        # Si llegamos aquí, al menos marcamos latido para el link
        self.state.link_heartbeat()


    def run(self):
        while not self._stop.is_set():
            client = None
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(self.host, port=self.port, username=self.username,
                               password=self.password, timeout=10.0)
                transport = client.get_transport()
                chan = transport.open_session()
                chan.exec_command(self.command)  # follow

                buff = b""
                while not self._stop.is_set() and not chan.exit_status_ready():
                    if chan.recv_ready():
                        chunk = chan.recv(4096)
                        if not chunk:
                            break
                        buff += chunk
                        while b"\n" in buff:
                            line, buff = buff.split(b"\n", 1)
                            try:
                                self._handle_line(line.decode("utf-8", errors="ignore"))
                            except Exception:
                                pass
                    else:
                        time.sleep(0.05)
            except (paramiko.SSHException, socket.error, TimeoutError):
                time.sleep(self.reconnect_sec)
            except Exception:
                time.sleep(self.reconnect_sec)
            finally:
                try:
                    if client:
                        client.close()
                except Exception:
                    pass

# =========================
# Utilidades de overlay
# =========================
def draw_signal_overlays(frame, cfg: dict, state: SemaforoState):
    primary = (cfg.get("primary_signal_id") or "").upper()
    only_primary = bool(cfg.get("show_only_primary", False))

    for s in cfg["signals"]:
        sid = s["id"].upper()
        if only_primary and sid != primary:
            continue

        pos = (int(s["overlay"]["x"]), int(s["overlay"]["y"]))
        sample = state.get_view(sid)
        color = COLOR_BGR.get(sample.state, COLOR_BGR["GRAY"])

        is_primary = (sid == primary)
        radius = 20 if is_primary else 14
        border = 3 if is_primary else 2

        # círculo
        cv2.circle(frame, pos, radius, color, thickness=-1)
        cv2.circle(frame, pos, radius, (0, 0, 0), thickness=border)

        # etiqueta
        label = sid
        if sample.icon:
            label += f" · {sample.icon}"
        if is_primary:
            label = "★ " + label

        cv2.putText(frame, label, (pos[0] + radius + 12, pos[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (pos[0] + radius + 12, pos[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def load_signals_config(path="signals_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
