"""
Recolector de Gestos para MouXe ML
══════════════════════════════════════════════════════

Sin límite de frames - grabas los que quieras.
Añade: ZOOM_IN, ZOOM_OUT (dos manos)

Usage:
    python recolectar.py

Controles:
    SPACE = Guardar secuencia
    S     = Siguiente gesto
    Q     = Salir
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# GESTOS (sin límite)
GESTOS_A_GRABAR = [
    ('MOVE', 'Índice levantado'),
    ('LEFT_CLICK', 'Pulgar + índice'),
    ('RIGHT_CLICK', 'Pulgar + corazón'),
    ('SCROLL', 'Índice + corazón'),
    ('FORWARD', 'Pulgar + anular'),
    ('BACK', 'Pulgar + meñique'),
    ('PUPPET', 'Todos juntos'),
    ('FIST', 'Puño cerrado'),
    ('PALM', 'Palma abierta'),
    ('ZOOM_IN', 'DOS MANOS - Abrir (alejar)'),
    ('ZOOM_OUT', 'DOS MANOS - Cerrar (acercar)'),
]

OUTPUT_FILE = "data/gestos_raw.npy"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def extract_features(landmarks_list):
    """Extrae features de una o dos manos"""
    features = []
    wrist = landmarks_list[0][0]
    
    # Primera mano
    for lm in landmarks_list[0]:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    
    # Segunda mano si existe
    if len(landmarks_list) > 1:
        wrist2 = landmarks_list[1][0]
        for lm in landmarks_list[1]:
            features.extend([lm.x - wrist2.x, lm.y - wrist2.y, lm.z - wrist2.z])
    else:
        features.extend([0.0] * 63)  # Relleno para segunda mano
    
    return np.array(features, dtype=np.float32)

def main():
    os.makedirs("data", exist_ok=True)
    
    # Cargar datos existentes
    datos = {nombre: [] for nombre, _ in GESTOS_A_GRABAR}
    if os.path.exists(OUTPUT_FILE):
        print(f"Cargando datos de {OUTPUT_FILE}...")
        try:
            data_existing = np.load(OUTPUT_FILE, allow_pickle=True).item()
            for nombre, _ in GESTOS_A_GRABAR:
                if nombre in data_existing:
                    datos[nombre] = list(data_existing[nombre])
            total = sum(len(v) for v in datos.values())
            print(f"✓ {total} secuencias cargadas")
        except:
            print("Archivo nuevo")
    
    print("\n" + "="*60)
    print("  RECOLECTOR MOUXE ML (SIN LÍMITE)")
    print("="*60)
    print("\nProgreso:")
    for nombre, desc in GESTOS_A_GRABAR:
        actual = len(datos[nombre])
        print(f"  {nombre:12} = {actual} secuencias")
    print("\n" + "-"*60)
    print("  SPACE = Guardar secuencia")
    print("  S     = Siguiente gesto")
    print("  Q     = Salir")
    print("-"*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Primer gesto incompleto
    gesto_idx = 0
    for i, (nombre, _) in enumerate(GESTOS_A_GRABAR):
        if len(datos[nombre]) < 50:  # Recomendado pero no obligatorio
            gesto_idx = i
            break
    
    buffer = []
    
    while cap.isOpened():
        nombre_actual, desc_actual = GESTOS_A_GRABAR[gesto_idx]
        count_actual = len(datos[nombre_actual])
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Gestos actual / siguiente
        next_idx = (gesto_idx + 1) % len(GESTOS_A_GRABAR)
        nombre_siguiente, desc_siguiente = GESTOS_A_GRABAR[next_idx]

        # UI
        cv2.rectangle(frame, (0, 0), (w, 140), (20, 20, 20), -1)
        cv2.putText(frame, f"GESTO: {nombre_actual}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, desc_actual, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, f"SIGUIENTE: {nombre_siguiente}", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, desc_siguiente, (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.rectangle(frame, (0, h-50), (w, h), (20, 20, 20), -1)
        
        if nombre_actual.startswith('ZOOM'):
            texto = f"Frames: {len(buffer)} | MANOS: {len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0}/2 | SPACE=grabar"
        else:
            texto = f"Frames: {len(buffer)} | {count_actual} secuencias | SPACE=grabar"
        
        cv2.putText(frame, texto, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detectar manos
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            
            # Extraer features
            landmarks_list = [h.landmark for h in result.multi_hand_landmarks]
            buffer.append(extract_features(landmarks_list))
        
        cv2.imshow("Recolector MouXe", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            buffer = []
            gesto_idx = (gesto_idx + 1) % len(GESTOS_A_GRABAR)
        elif key == ord(' '):
            if len(buffer) > 0:
                secuencia = np.array(buffer, dtype=np.float32)
                datos[nombre_actual].append(secuencia)
                np.save(OUTPUT_FILE, datos)
                print(f"  ✓ {nombre_actual}: {count_actual+1} secuencias ({len(buffer)} frames)")
                buffer = []
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Guardado en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
