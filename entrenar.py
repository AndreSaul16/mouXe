"""
Entrenador de Modelo BiLSTM para MouXe
══════════════════════════════════════════════════════

Este script entrena el modelo BiLSTM con los datos recolectados.

Usage:
    python entrenar.py

Requirements:
    - Haber ejecutado recolectar.py primero
    -Tener datos en data/gestos_raw.npy
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dropout, Dense
)
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

DATA_FILE = "data/gestos_raw.npy"
MODEL_FILE = "models/gesture_model.h5"
LABELS_FILE = "models/gesture_labels.pkl"

BUFFER_SIZE = 30    # Frames por sample (debe coincidir con recolector)
NUM_FEATURES = 126   # 21 landmarks * 3 coordenadas * 2 manos
NUM_CLASSES = 11    # MOVE, LEFT_CLICK, RIGHT_CLICK, SCROLL, FORWARD, BACK, PUPPET, FIST, PALM, ZOOM_IN, ZOOM_OUT

EPOCHS = 50
BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(num_classes=NUM_CLASSES, input_shape=(BUFFER_SIZE, NUM_FEATURES)):
    """
    Crea el modelo BiLSTM.
    Arquitectura basada en el paper original.
    """
    model = Sequential([
        # Input
        tf.keras.layers.Input(shape=input_shape),
        
        # BiLSTM Layer 1
        Bidirectional(LSTM(128, return_sequences=True, activation='sigmoid')),
        Dropout(0.1),
        
        # BiLSTM Layer 2
        Bidirectional(LSTM(64, return_sequences=True, activation='sigmoid')),
        Dropout(0.1),
        
        # BiLSTM Layer 3
        Bidirectional(LSTM(32, return_sequences=False, activation='sigmoid')),
        
        # Dense layers
        Dense(128, activation='sigmoid'),
        Dropout(0.1),
        Dense(64, activation='sigmoid'),
        Dropout(0.1),
        
        # Output
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ENTRENADOR DE MODELO BILSTM PARA MOUXE          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Verificar que existen los datos
    if not os.path.exists(DATA_FILE):
        print(f"✗ ERROR: No se encontró {DATA_FILE}")
        print("  Ejecuta primero: python recolectar.py")
        sys.exit(1)
    
    # Cargar datos (soporta formato de recolectar.py y/o formato entrenar.py)
    print(f"Cargando datos de {DATA_FILE}...")
    data = np.load(DATA_FILE, allow_pickle=True).item()

    if all(k in data for k in ('X', 'y', 'labels')):
        # Formato ya procesado
        X = data['X']
        y = data['y']
        labels = data['labels']
        print(f"✓ Datos cargados (formato procesado)")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Labels: {labels}")
        print()
    else:
        # Formato crudo de recolectar.py: {gesto: [secuencia, ...], ...}
        print("✓ Datos cargados (formato crudo) - creando X/y/labels")

        # Asegurar orden consistente de labels
        GESTOS_A_GRABAR = [
            'MOVE', 'LEFT_CLICK', 'RIGHT_CLICK', 'SCROLL',
            'FORWARD', 'BACK', 'PUPPET', 'FIST',
            'PALM', 'ZOOM_IN', 'ZOOM_OUT',
        ]

        labels = GESTOS_A_GRABAR
        X = []
        y = []
        for idx, gesto in enumerate(labels):
            secuencias = data.get(gesto, [])
            for seq in secuencias:
                X.append(seq)
                y.append(idx)

        X = np.array(X, dtype=object)
        y = np.array(y, dtype=np.int32)
        print(f"  Gestos encontrados: {len(labels)}")
        print(f"  Secuencias totales: {len(X)}")
        print()
    
    # Verificar dimensiones (ahora pueden variar)
    min_len = min(len(x) for x in X)
    max_len = max(len(x) for x in X)
    print(f"  Longitud secuencias: min={min_len}, max={max_len}")
    
    # Detectar dimensión máxima de features (63 o 126)
    max_features = max(np.array(s).shape[1] for g in data for s in data[g] if len(s) > 0)
    print(f"  Dimensión de features detectada: {max_features}")
    
    # Padding todas las secuencias a la misma longitud y dimensión de features
    X_fixed = []
    for seq in X:
        seq = np.array(seq, dtype=np.float32)
        # 1. Ajustar features (de 63 a max_features)
        if seq.shape[1] < max_features:
            padding_features = np.zeros((seq.shape[0], max_features - seq.shape[1]), dtype=np.float32)
            seq = np.hstack([seq, padding_features])
        
        # 2. Ajustar longitud temporal (max_len)
        if len(seq) < max_len:
            padding_time = np.zeros((max_len - len(seq), max_features), dtype=np.float32)
            seq = np.vstack([seq, padding_time])
        else:
            seq = seq[:max_len]
            
        X_fixed.append(seq)
    
    X = np.array(X_fixed, dtype=np.float32)
    print(f"  X shape final: {X.shape}")
    
    # Convertir labels a one-hot
    y_hot = tf.keras.utils.to_categorical(y, num_classes=len(labels))
    
    # Crear modelo
    print("Creando modelo BiLSTM...")
    model = create_model(num_classes=len(labels), input_shape=(X.shape[1], X.shape[2]))
    model.summary()
    print()
    
    # Entrenar
    print("Iniciando entrenamiento...")
    print("=" * 50)
    
    history = model.fit(
        X, y_hot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    print("=" * 50)
    print()
    
    # Evaluación final
    val_loss, val_acc = model.evaluate(X, y_hot, verbose=0)
    print(f"Precisión final (training): {val_acc:.2%}")
    print()
    
    # Guardar modelo
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    model.save(MODEL_FILE)
    print(f"✓ Modelo guardado en {MODEL_FILE}")
    
    # Guardar labels
    import pickle
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(labels, f)
    print(f"✓ Labels guardados en {LABELS_FILE}")
    
    print()
    print("══════════════════════════════════════════════════════════")
    print("✓ ENTRENAMIENTO COMPLETADO!")
    print("══════════════════════════════════════════════════════════")
    print()
    print("Ahora puedes ejecutar: python mouxe.py")
    print("El modelo ML se cargará automáticamente.")


if __name__ == "__main__":
    main()
