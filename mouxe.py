"""
MouXe - Control tu ordenador con una sola mano
═══════════════════════════════════════════════
  UNA SOLA MANO (izquierda o derecha, ambas funcionan igual)

    ☝️  Índice levantado               → Mover cursor
    🤏  Pulgar+índice                   → Clic izquierdo (sostener >0.5s = hold/arrastrar)
    🖕  Pulgar+corazón                  → Clic derecho
    💍  Pulgar+anular                   → Avanzar (botón lateral mouse)
    🤙  Pulgar+meñique                  → Retroceder (botón lateral mouse)
    ✌️  Índice+corazón levantados       → Scroll (mueve arriba/abajo)
    🤌  Todos juntos (marioneta)        → Scroll hold (botón central mantenido)

  Presiona Q en la ventana para salir.
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import time
import ctypes

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
CAMARA            = 0         # 0 = webcam principal
NUM_PANTALLAS     = 2         # 1 = un monitor, 2 = doble pantalla (ajusta ganancia)
SUAVIZADO         = 0.12      # más bajo = más suave/lento, más alto = más ágil
PINCH_RATIO       = 0.35      # pellizco = distancia < este % del tamaño de la mano
PINCH_RELEASE     = 0.45      # soltar = distancia > este % (histéresis)
FREEZE_RATIO      = 0.55      # cursor se congela cuando el pulgar se acerca a cualquier dedo
FINGER_UP_MARGIN  = 0.01      # margen para considerar dedo "levantado"
BOUNCE_COOLDOWN   = 0.08      # segundos mínimos entre mouseDown (anti-rebote)
SCROLL_SPEED      = 10        # velocidad de scroll
HOLD_VISUAL_TIME  = 0.5       # segundos para mostrar indicador [HOLD] en HUD
# Zonas activas de la cámara — se ajustan automáticamente por NUM_PANTALLAS
# Con más pantallas las zonas se estrechan → más ganancia → menos movimiento necesario
ZONE_X_MIN        = 0.50 - 0.35 / NUM_PANTALLAS   # 1 pantalla: 0.15  |  2 pantallas: 0.325
ZONE_X_MAX        = 0.50 + 0.35 / NUM_PANTALLAS   # 1 pantalla: 0.85  |  2 pantallas: 0.675
ZONE_Y_MIN        = 0.10
ZONE_Y_MAX        = 0.75
MIRROR            = True      # espejo horizontal
MOSTRAR_CAMARA    = True      # mostrar ventana de debug
# ──────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

SCREEN_W, SCREEN_H = pyautogui.size()

# Índices de landmarks
PULGAR_TIP  = 4;  PULGAR_IP   = 3
INDICE_TIP  = 8;  INDICE_PIP  = 6
CORAZON_TIP = 12; CORAZON_PIP = 10
ANULAR_TIP  = 16; ANULAR_PIP  = 14
MENIQUE_TIP = 20; MENIQUE_PIP = 18
WRIST       = 0
MIDDLE_MCP  = 9


def hand_size(lm):
    """Distancia muñeca → base dedo corazón (referencia de tamaño)."""
    return math.hypot(lm[WRIST].x - lm[MIDDLE_MCP].x,
                      lm[WRIST].y - lm[MIDDLE_MCP].y)


def dist_norm(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y + FINGER_UP_MARGIN


def lerp(a, b, t):
    return a + (b - a) * t


# ─── BOTÓN ESTÁNDAR CON HOLD ─────────────────────────────────────────────────

class HoldButton:
    """mouseDown en press, mouseUp en release. Soporta tap y hold."""
    def __init__(self, button: str):
        self.button    = button
        self.held      = False
        self.last_down = 0

    def update(self, pressed: bool):
        now = time.time()
        if pressed and not self.held:
            if (now - self.last_down) > BOUNCE_COOLDOWN:
                pyautogui.mouseDown(button=self.button)
                self.held      = True
                self.last_down = now
        elif not pressed and self.held:
            pyautogui.mouseUp(button=self.button)
            self.held = False

    def release(self):
        if self.held:
            pyautogui.mouseUp(button=self.button)
            self.held = False


# ─── BOTÓN EXTRA: AVANZAR / RETROCEDER (Win32 API) ───────────────────────────

MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP   = 0x0100
XBUTTON1          = 0x0001   # Retroceder
XBUTTON2          = 0x0002   # Avanzar


class XButton:
    """Botón lateral del ratón (XButton1/XButton2) vía Win32 mouse_event."""
    def __init__(self, xbutton: int):
        self.xbutton   = xbutton
        self.held      = False
        self.last_down = 0

    def update(self, pressed: bool):
        now = time.time()
        if pressed and not self.held:
            if (now - self.last_down) > BOUNCE_COOLDOWN:
                ctypes.windll.user32.mouse_event(
                    MOUSEEVENTF_XDOWN, 0, 0, self.xbutton, 0)
                self.held      = True
                self.last_down = now
        elif not pressed and self.held:
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_XUP, 0, 0, self.xbutton, 0)
            self.held = False

    def release(self):
        if self.held:
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_XUP, 0, 0, self.xbutton, 0)
            self.held = False


# ─── CONTROLADOR PRINCIPAL ────────────────────────────────────────────────────

class MouXe:
    def __init__(self):
        self.cx = SCREEN_W / 2
        self.cy = SCREEN_H / 2

        # Botones
        self.btn_izq  = HoldButton("left")
        self.btn_der  = HoldButton("right")
        self.btn_mid  = HoldButton("middle")
        self.btn_fwd  = XButton(XBUTTON2)
        self.btn_back = XButton(XBUTTON1)

        # Histéresis de pellizcos {clave: activo}
        self._pinch = {'pi': False, 'pm': False, 'pa': False, 'pk': False}

        # Scroll
        self.scroll_ref  = None
        self.scroll_mode = False

    def _hyst(self, key, ratio):
        """Histéresis: umbral bajo para activar, alto para soltar."""
        if self._pinch[key]:
            self._pinch[key] = ratio < PINCH_RELEASE
        else:
            self._pinch[key] = ratio < PINCH_RATIO
        return self._pinch[key]

    def _mover_cursor(self, raw_x, raw_y):
        """Mapea coordenadas normalizadas (0-1) → pantalla, con suavizado."""
        nx = (raw_x - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)
        ny = (raw_y - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        tx = nx * SCREEN_W
        ty = ny * SCREEN_H
        self.cx = lerp(self.cx, tx, SUAVIZADO)
        self.cy = lerp(self.cy, ty, SUAVIZADO)
        pyautogui.moveTo(int(self.cx), int(self.cy))

    def procesar(self, lm, w, h):
        hs = hand_size(lm) or 0.001

        # ── Ratios de pellizco (normalizados por tamaño de mano) ──
        r_pi = dist_norm(lm[PULGAR_TIP], lm[INDICE_TIP])  / hs
        r_pm = dist_norm(lm[PULGAR_TIP], lm[CORAZON_TIP]) / hs
        r_pa = dist_norm(lm[PULGAR_TIP], lm[ANULAR_TIP])  / hs
        r_pk = dist_norm(lm[PULGAR_TIP], lm[MENIQUE_TIP]) / hs

        # ── Histéresis por dedo ──
        pinch_pi = self._hyst('pi', r_pi)
        pinch_pm = self._hyst('pm', r_pm)
        pinch_pa = self._hyst('pa', r_pa)
        pinch_pk = self._hyst('pk', r_pk)

        # ── Dedos levantados ──
        i_up = finger_up(lm, INDICE_TIP, INDICE_PIP)
        m_up = finger_up(lm, CORAZON_TIP, CORAZON_PIP)

        # ── Congelación de cursor: si el pulgar se acerca a CUALQUIER dedo ──
        min_ratio = min(r_pi, r_pm, r_pa, r_pk)
        approaching = min_ratio < FREEZE_RATIO

        # ── Determinar gesto activo ──
        puppet = pinch_pi and pinch_pm and pinch_pa and pinch_pk
        gesture = None

        if puppet:
            gesture = 'puppet'
        elif pinch_pi or pinch_pm or pinch_pa or pinch_pk:
            # Si varios dedos pellizcan, el más cercano al pulgar gana
            candidates = []
            if pinch_pi: candidates.append(('left_click', r_pi))
            if pinch_pm: candidates.append(('right_click', r_pm))
            if pinch_pa: candidates.append(('forward', r_pa))
            if pinch_pk: candidates.append(('back', r_pk))
            gesture = min(candidates, key=lambda x: x[1])[0]
        elif i_up and m_up:
            gesture = 'scroll'
        elif i_up:
            gesture = 'move'

        # ── Actualizar TODOS los botones según gesto activo ──
        self.btn_mid.update(gesture == 'puppet')
        self.btn_izq.update(gesture == 'left_click')
        self.btn_der.update(gesture == 'right_click')
        self.btn_fwd.update(gesture == 'forward')
        self.btn_back.update(gesture == 'back')

        # ── Scroll (índice + corazón levantados) ──
        if gesture == 'scroll':
            if not self.scroll_mode:
                self.scroll_ref  = lm[INDICE_TIP].y
                self.scroll_mode = True
            else:
                dy = self.scroll_ref - lm[INDICE_TIP].y
                if abs(dy) > 0.01:
                    amount = int(dy * SCROLL_SPEED * 8)
                    if amount != 0:
                        pyautogui.scroll(amount)
            return "SCROLL"
        else:
            self.scroll_mode = False
            self.scroll_ref  = None

        # ── Mover cursor (solo índice levantado) ──
        # Se congela cuando el pulgar se acerca a cualquier dedo (pre-pellizco)
        if gesture == 'move':
            if not approaching:
                self._mover_cursor(lm[INDICE_TIP].x, lm[INDICE_TIP].y)
                return "MOVIENDO"
            return "APUNTANDO"

        # ── Clic izquierdo: drag solo tras HOLD ──
        if gesture == 'left_click':
            held_time = time.time() - self.btn_izq.last_down
            if held_time > HOLD_VISUAL_TIME:
                mid_x = (lm[PULGAR_TIP].x + lm[INDICE_TIP].x) / 2
                mid_y = (lm[PULGAR_TIP].y + lm[INDICE_TIP].y) / 2
                self._mover_cursor(mid_x, mid_y)
                return "CLICK IZQ [HOLD]"
            return "CLICK IZQ"

        if gesture == 'right_click':
            return "CLICK DER"
        if gesture == 'forward':
            return "AVANZAR"
        if gesture == 'back':
            return "RETROCEDER"
        if gesture == 'puppet':
            return "SCROLL HOLD"

        return "INACTIVA"

    def reset(self):
        """Resetear estado cuando la mano desaparece."""
        self.liberar_todo()
        self.scroll_mode = False
        self.scroll_ref  = None
        for k in self._pinch:
            self._pinch[k] = False

    def liberar_todo(self):
        """Soltar todos los botones que estén pulsados."""
        self.btn_izq.release()
        self.btn_der.release()
        self.btn_mid.release()
        self.btn_fwd.release()
        self.btn_back.release()


# ─── HUD ──────────────────────────────────────────────────────────────────────

COLORES = {
    "MOVIENDO":     (0,   255, 120),
    "APUNTANDO":    (0,   180,  80),
    "CLICK IZQ":    (50,  220, 255),
    "CLICK DER":    (255, 120,  30),
    "SCROLL HOLD":  (180, 100, 255),
    "SCROLL":       (200,  50, 255),
    "AVANZAR":      (100, 255, 100),
    "RETROCEDER":   (100, 100, 255),
}


def color_estado(estado):
    for key, col in COLORES.items():
        if estado.startswith(key):
            return col
    return (120, 120, 120)


def render_hud(frame, estado):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (fw, 48), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    col = color_estado(estado)
    cv2.putText(frame, estado, (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    mouxe = MouXe()
    cap   = cv2.VideoCapture(CAMARA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.45,
    ) as hands:

        print("MouXe activo — presiona Q en la ventana para salir")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            if MIRROR:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            estado = "INACTIVA"

            if result.multi_hand_landmarks:
                lm_obj = result.multi_hand_landmarks[0]
                estado = mouxe.procesar(lm_obj.landmark, fw, fh)

                if MOSTRAR_CAMARA:
                    mp_drawing.draw_landmarks(
                        frame, lm_obj,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
            else:
                mouxe.reset()

            if MOSTRAR_CAMARA:
                render_hud(frame, estado)
                cv2.imshow("MouXe", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    mouxe.liberar_todo()
    cap.release()
    cv2.destroyAllWindows()
    print("MouXe cerrado.")


if __name__ == "__main__":
    main()
