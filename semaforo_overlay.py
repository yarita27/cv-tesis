import json, time, threading, re
from dataclasses import dataclass, field
from typing import Dict, Optional
import paho.mqtt.client as mqtt
import cv2

# --------------------------
# Utilidades y estructuras
# --------------------------
COLOR_BGR = {
    "RED":    (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN":  (0, 255, 0),
    "GRAY":   (128, 128, 128)
}

PAYLOAD_RE = re.compile(r"^\s*(RED|YELLOW|GREEN)\s+ICON:([A-Z]+)\s*$", re.IGNORECASE)

@dataclass
class SignalSample:
    state: str = "GRAY"
    icon: Optional[str] = None
    ts: float = field(default_factory=time.time)

class SemaforoState:
    """Estado thread-safe de varios semáforos."""
    def __init__(self, stale_after_sec: float = 5.0):
        self._lock = threading.Lock()
        self._data: Dict[str, SignalSample] = {}
        self._stale_after = stale_after_sec

    def bootstrap(self, snapshot: Dict[str, Dict[str, str]]):
        with self._lock:
            now = time.time()
            for sid, info in snapshot.items():
                st = info.get("state", "GRAY").upper()
                ic = info.get("icon")
                self._data[sid] = SignalSample(state=st, icon=ic, ts=now)

    def update_from_mqtt(self, sid: str, payload: str):
        m = PAYLOAD_RE.match(payload)
        if not m:
            # Payload no reconocido -> no pisar estado válido anterior
            return
        state = m.group(1).upper()
        icon = m.group(2).upper()
        with self._lock:
            self._data[sid] = SignalSample(state=state, icon=icon, ts=time.time())

    def get_view(self, sid: str) -> SignalSample:
        with self._lock:
            s = self._data.get(sid, SignalSample())
            # Marcar gris si está "stale"
            if (time.time() - s.ts) > self._stale_after:
                return SignalSample(state="GRAY", icon=s.icon, ts=s.ts)
            return s

class MQTTSemaforoBridge(threading.Thread):
    """Suscriptor MQTT que alimenta SemaforoState."""
    def __init__(self, host: str, port: int, topic_prefix: str, signals_ids, state: SemaforoState, username="", password=""):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.topic_prefix = topic_prefix.rstrip('/') + '/'
        self.signals_ids = list(signals_ids)
        self.state = state
        self.client = mqtt.Client()
        if username or password:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        # Suscribirse a cada S* explicitamente
        for sid in self.signals_ids:
            t = f"{self.topic_prefix}{sid}"
            client.subscribe(t, qos=1)

    def _on_message(self, client, userdata, msg):
        # topic ej: esp32/semaforos/S1 -> sid = S1
        sid = msg.topic.split('/')[-1].upper()
        payload = msg.payload.decode('utf-8', errors='ignore')
        self.state.update_from_mqtt(sid, payload)

    def run(self):
        self.client.connect(self.host, self.port, 60)
        self.client.loop_forever()

# --------------------------
# Overlay en OpenCV
# --------------------------
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

        # Estilo distinto si es el semáforo principal
        is_primary = (sid == primary)
        radius = 20 if is_primary else 14
        border = 3 if is_primary else 2

        # Círculo
        cv2.circle(frame, pos, radius, color, thickness=-1)
        cv2.circle(frame, pos, radius, (0, 0, 0), thickness=border)

        # Etiqueta: añade estrella si es el principal
        label = sid
        if sample.icon:
            label += f" · {sample.icon}"
        if is_primary:
            label = "★ " + label  # marca visual del local

        cv2.putText(frame, label, (pos[0] + radius + 12, pos[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (pos[0] + radius + 12, pos[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def load_signals_config(path="signals_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
