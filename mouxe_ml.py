"""
MouXe ML - Módulo de Reconocimiento de Gestos con BiLSTM
════════════════════════════════════════════════════════

Este módulo añade aprendizaje automático al control de gestos de MouXe.
Usa un buffer de frames + BiLSTM para mayor robustez en el reconocimiento.

La integración funciona así:
- Buffer de 30 frames de landmarks (secuencia temporal)
- Predicción del gesto mediante BiLSTM
- Fallback a lógica clásica de MouXe si el modelo no está disponible

Arquitectura BiLSTM basada en:
https://github.com/gspagare/Real-Time-Gesture-Recognition-Using-Mediapipe-and-BiLSTM
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import pickle

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

BUFFER_SIZE = 30              # Frames a acumular para predicción
NUM_FEATURES = 63             # 21 landmarks × 3 coordenadas (x, y, z)
MODEL_PATH = "models/gesture_model.h5"
LABELS_PATH = "models/gesture_labels.pkl"
MIN_CONFIDENCE = 0.85         # Threshold mínimo para aceptar predicción

# ═══════════════════════════════════════════════════════════════════════════════
# MAPA DE GESTOS - Coincide con los gestos de MouXe
# ═══════════════════════════════════════════════════════════════════════════════

GESTO_A_ACCION = {
    0: 'MOVE',          # Índice levantado - mover cursor
    1: 'LEFT_CLICK',    # Pulgar + índice - clic izquierdo
    2: 'RIGHT_CLICK',   # Pulgar + corazón - clic derecho
    3: 'SCROLL',        # Índice + corazón levantados - scroll
    4: 'FORWARD',       # Pulgar + anular - avanzar
    5: 'BACK',          # Pulgar + meñique - retroceder
    6: 'PUPPET',        # Todos juntos - scroll hold
    7: 'FIST',          # Puño - drag/hold
    8: 'PALM',         # Palma abierta - inactivo
    9: 'ZOOM_IN',      # Dos manos abriéndose - zoom in
    10: 'ZOOM_OUT',    # Dos manos cerrándose - zoom out
}

# Gesto por defecto (cuando no se detecta mano)
DEFAULT_GESTURE = 'INACTIVE'

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_hand_features(landmarks):
    """
    Extrae features (x, y, z) de los 21 landmarks de la mano.
    
    Args:
        landmarks: Lista de 21 landmark objects de MediaPipe
        
    Returns:
        np.array de shape (63,) con los features normalizados
    """
    features = []
    
    # Muñeca como referencia (índice 0)
    wrist = landmarks[0]
    
    for lm in landmarks:
        # Coordenadas relativas a la muñeca para invarianza de posición
        features.append(lm.x - wrist.x)
        features.append(lm.y - wrist.y)
        features.append(lm.z - wrist.z)
    
    return np.array(features, dtype=np.float32)


def normalize_features(features):
    """
    Normaliza features al rango [0, 1] basándose en valores típicos.
    Esto ayuda al modelo a converger más rápido.
    """
    # Valores típicos de coordenadas normalizadas de MediaPipe
    # x e y están en [0, 1], z es relativo
    normalized = features.copy()
    
    # Las primeras 42 valores son x,y (0-1 ya normalizados)
    # Los últimos 21 son z (pueden ser negativos, típicamente -0.1 a 0.1)
    for i in range(42, 63):
        # Normalizar z: asume rango [-0.15, 0.15]
        normalized[i] = (normalized[i] + 0.15) / 0.3
    
    return normalized

# ═══════════════════════════════════════════════════════════════════════════════
# BUFFER DE SECUENCIAS TEMPORALES
# ═══════════════════════════════════════════════════════════════════════════════

class GestureBuffer:
    """
    Buffer circular que almacena los últimos N frames de landmarks.
    Proporciona la secuencia temporal para el modelo BiLSTM.
    """
    
    def __init__(self, maxlen=BUFFER_SIZE):
        self.buffer = deque(maxlen=maxlen)
        self.is_full = False
    
    def add_frame(self, landmarks):
        """Añade un frame de landmarks al buffer."""
        if landmarks is not None:
            features = extract_hand_features(landmarks)
            features = normalize_features(features)
            self.buffer.append(features)
            
            if len(self.buffer) >= BUFFER_SIZE:
                self.is_full = True
    
    def get_sequence(self):
        """
        Retorna la secuencia actual como array numpy.
        Shape: (30, 63) si está lleno, (n, 63) si no.
        """
        # Rellena con ceros si no está lleno (padding)
        sequence = list(self.buffer)
        
        if len(sequence) < BUFFER_SIZE:
            # Padding al inicio con ceros
            padding = [np.zeros(NUM_FEATURES, dtype=np.float32)] * (BUFFER_SIZE - len(sequence))
            sequence = padding + sequence
        
        return np.array(sequence, dtype=np.float32)
    
    def reset(self):
        """Limpia el buffer."""
        self.buffer.clear()
        self.is_full = False
    
    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO BiLSTM
# ═══════════════════════════════════════════════════════════════════════════════

def create_bilstm_model(num_classes=11):
    """
    Crea el modelo BiLSTM basado en la arquitectura del repo de gspagare.
    
    Input Shape: (30, 63) - 30 frames, 63 features por frame
    Output: num_classes acciones
    
    Arquitectura:
    - BiLSTM (128 unidades, return_sequences=True)
    - Dropout (0.1)
    - BiLSTM (64 unidades, return_sequences=True)
    - Dropout (0.1)
    - BiLSTM (32 unidades)
    - Dense (128) -> Dropout -> Dense (64) -> Dense (num_classes, softmax)
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(BUFFER_SIZE, NUM_FEATURES)),
        
        # BiLSTM Layer 1
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, activation='sigmoid')
        ),
        tf.keras.layers.Dropout(0.1),
        
        # BiLSTM Layer 2
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, activation='sigmoid')
        ),
        tf.keras.layers.Dropout(0.1),
        
        # BiLSTM Layer 3
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=False, activation='sigmoid')
        ),
        
        # Dense layers
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class GestureClassifier:
    """
    Clasificador de gestos basado en BiLSTM.
    Maneja carga del modelo, predicción y fallback.
    """
    
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = GESTO_A_ACCION  # Default labels
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Intenta cargar el modelo entrenado."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                
                # Cargar labels si existen
                if os.path.exists(self.labels_path):
                    with open(self.labels_path, 'rb') as f:
                        loaded = pickle.load(f)

                    # Normalize labels into a mapping index -> label
                    if isinstance(loaded, dict):
                        # Common formats: {idx: label} or {label: idx}
                        if all(isinstance(k, int) for k in loaded.keys()):
                            self.labels = loaded
                        elif all(isinstance(v, int) for v in loaded.values()):
                            self.labels = {v: k for k, v in loaded.items()}
                        else:
                            self.labels = loaded
                    elif isinstance(loaded, (list, tuple)):
                        self.labels = {i: label for i, label in enumerate(loaded)}
                    else:
                        self.labels = loaded

                self.is_loaded = True
                print(f"✓ Modelo ML cargado desde {self.model_path}")
            except Exception as e:
                print(f"✗ Error cargando modelo: {e}")
                self.is_loaded = False
        else:
            print(f"⚠ Modelo no encontrado en {self.model_path}")
            print("  Usando lógica clásica de MouXe como fallback")
            self.is_loaded = False
    
    def predict(self, sequence):
        """
        Predice el gesto a partir de una secuencia de frames.
        
        Args:
            sequence: Array numpy de shape (30, 63)
            
        Returns:
            Tupla (gesto, confianza, siguiente_gesto, siguiente_confianza) o (None, 0, None, 0) si no se puede predecir
        """
        if not self.is_loaded or self.model is None:
            return None, 0.0, None, 0.0
        
        # Reshape para el modelo: (1, 30, 63)
        sequence = sequence.reshape(1, BUFFER_SIZE, NUM_FEATURES)
        
        # Predicción
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Obtener los índices ordenados por probabilidad (mayor a menor)
        sorted_indices = np.argsort(predictions)[::-1]
        
        # Predicción principal
        predicted_idx = sorted_indices[0]
        confidence = predictions[predicted_idx]
        
        # Segunda predicción más probable
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else -1
        second_confidence = predictions[second_idx] if second_idx >= 0 else 0.0
        
        # Verificar threshold de confianza para la predicción principal
        if confidence >= MIN_CONFIDENCE:
            gesto = self.labels.get(predicted_idx, DEFAULT_GESTURE)
            siguiente = self.labels.get(second_idx, None) if second_idx >= 0 else None
            return gesto, float(confidence), siguiente, float(second_confidence)
        
        return None, float(confidence), self.labels.get(second_idx, None) if second_idx >= 0 else None, float(second_confidence)
    
    def save_model(self, save_path=None):
        """Guarda el modelo entrenado."""
        if self.model is not None:
            path = save_path or self.model_path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            print(f"✓ Modelo guardado en {path}")
    
    def save_labels(self, save_path=None):
        """Guarda los labels del modelo."""
        path = save_path or self.labels_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.labels, f)
        print(f"✓ Labels guardados en {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# HERRAMIENTAS DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

class GestureRecorder:
    """
    Graba secuencias de gestos para entrenamiento.
    Usa la misma arquitectura de buffer que el clasificador.
    """
    
    def __init__(self, gesture_name, num_samples=100):
        self.gesture_name = gesture_name
        self.num_samples = num_samples
        self.samples = []
        self.buffer = GestureBuffer()
        self.recording = False
        self.sample_count = 0
    
    def start_recording(self):
        """Inicia la grabación de samples."""
        self.recording = True
        self.sample_count = 0
        self.samples = []
        print(f"🎬 Grabando gesto '{self.gesture_name}'...")
        print(f"   Necesitas {self.num_samples} samples. Mantén el gesto.")
    
    def add_sample(self, landmarks):
        """Añade un frame al sample actual."""
        if not self.recording:
            return False
        
        self.buffer.add_frame(landmarks)
        
        # Cuando el buffer está lleno, guardar como sample
        if self.buffer.is_full:
            sequence = self.buffer.get_sequence()
            self.samples.append(sequence)
            self.sample_count += 1
            self.buffer.reset()
            
            print(f"   Sample {self.sample_count}/{self.num_samples}")
            
            if self.sample_count >= self.num_samples:
                self.stop_recording()
                return False
        
        return True
    
    def stop_recording(self):
        """Detiene la grabación."""
        self.recording = False
        print(f"✓ Grabación completada: {len(self.samples)} samples")
    
    def get_samples(self):
        """Retorna los samples grabados."""
        return np.array(self.samples)


def train_model(X_train, y_train, num_classes=10, epochs=50, batch_size=32):
    """
    Entrena el modelo BiLSTM con los datos proporcionados.
    
    Args:
        X_train: Datos de entrenamiento, shape (n_samples, 30, 63)
        y_train: Labels, shape (n_samples,)
        num_classes: Número de clases
        epochs: Épocas de entrenamiento
        batch_size: Batch size
        
    Returns:
        Modelo entrenado
    """
    # Convertir labels a one-hot
    y_train_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    
    # Crear modelo
    model = create_bilstm_model(num_classes)
    
    # Entrenar
    model.fit(
        X_train, y_train_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO - Grabación de nuevos gestos
# ═══════════════════════════════════════════════════════════════════════════════

def demo_record_gesture():
    """
    Demo: Cómo grabar un nuevo gesto para entrenamiento.
    """
    import cv2
    import mediapipe as mp
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    recorder = GestureRecorder("MI_NUEVO_GESTO", num_samples=50)
    recorder.start_recording()
    
    cap = cv2.VideoCapture(0)
    
    while recorder.recording:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            recorder.add_sample(landmarks)
        
        cv2.putText(frame, f"Grabando: {recorder.sample_count}/{recorder.num_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Grabando gesto", frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Obtener samples
    samples = recorder.get_samples()
    print(f"Shape de samples: {samples.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# PRUEBA RÁPIDA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test de creación de modelo
    print("🧪 Probando creación de modelo BiLSTM...")
    model = create_bilstm_model(num_classes=10)
    model.summary()
    
    # Test de buffer
    print("\n🧪 Probando GestureBuffer...")
    buffer = GestureBuffer()
    
    # Simular landmarks dummy
    class DummyLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    dummy_landmarks = [DummyLandmark(i*0.01, i*0.02, i*0.001) for i in range(21)]
    
    for _ in range(35):
        buffer.add_frame(dummy_landmarks)
    
    print(f"   Buffer lleno: {buffer.is_full}")
    print(f"   Secuencia shape: {buffer.get_sequence().shape}")
    
    # Test de clasificador
    print("\n🧪 Probando GestureClassifier...")
    classifier = GestureClassifier()
    print(f"   Modelo cargado: {classifier.is_loaded}")
