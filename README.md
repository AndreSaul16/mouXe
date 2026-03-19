# MouXe

Controla tu ordenador con una sola mano usando la webcam. Sin hardware extra, sin sensores, solo tu mano y una cámara.

MouXe usa **MediaPipe** para detectar la mano en tiempo real y traduce gestos naturales en acciones del ratón: mover el cursor, clic izquierdo/derecho, scroll, arrastrar, avanzar y retroceder en el navegador.

---

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/AndreSaul16/mouXe.git
cd mouXe

# Instalar dependencias
pip install -r requirements.txt
```

### Requisitos
- Python 3.8+
- Windows (los botones avanzar/retroceder usan Win32 API)
- Webcam

> **Nota:** Requiere `mediapipe==0.10.9`. Versiones 0.10.10+ eliminaron la API `mp.solutions` y no son compatibles.

---

## 📖 Uso

### Modo Básico (Clásico)
```bash
python mouxe.py
```
Presiona **Q** en la ventana de la cámara para salir.

### Modo ML (Aprendizaje Automático)
Para usar el reconocimiento de gestos con BiLSTM:

1. **Recolectar datos**:
   ```bash
   python recolectar.py
   ```
   Controles:
   - `SPACE` = Guardar secuencia del gesto actual
   - `S` = Siguiente gesto
   - `Q` = Salir

2. **Entrenar modelo**:
   ```bash
   python entrenar.py
   ```

3. **Ejecutar con ML**:
   ```bash
   python mouxe.py
   ```
   El modelo ML se cargará automáticamente si está disponible.

---

## 🎮 Controles

Funciona igual con mano izquierda o derecha.

| Gesto | Acción |
|-------|--------|
| ☝️ Índice levantado | Mover cursor |
| 🤏 Pulgar + índice | Clic izquierdo (mantener >0.5s = hold/arrastrar) |
| 🖕 Pulgar + corazón | Clic derecho |
| 💍 Pulgar + anular | Avanzar (botón lateral del mouse) |
| 🤙 Pulgar + meñique | Retroceder (botón lateral del mouse) |
| ✌️ Índice + corazón levantados | Scroll (mover arriba/abajo) |
| 🤌 Todos juntos (marioneta) | Scroll hold (botón central mantenido) |

---

## 🧠 Versión ML (BiLSTM)

El módulo de aprendizaje automático usa una red neuronal **BiLSTM** para un reconocimiento de gestos más robusto.

### Características:
- **Buffer temporal**: Analiza los últimos 30 frames para contexto temporal
- **BiLSTM**: Red neuronal bidireccional para clasificación de gestos
- **Fallback automático**: Si no hay modelo entrenado, usa la lógica clásica
- **Threshold de confianza**: Solo usa predicción ML si confidence >= 85%
- **Soporte dos manos**: ZOOM_IN y ZOOM_OUT

### Gestos soportados en ML:
1. MOVE - Índice levantado
2. LEFT_CLICK - Pulgar + índice
3. RIGHT_CLICK - Pulgar + corazón
4. SCROLL - Índice + corazón
5. FORWARD - Pulgar + anular
6. BACK - Pulgar + meñique
7. PUPPET - Todos juntos
8. FIST - Puño cerrado
9. PALM - Palma abierta
10. ZOOM_IN - Dos manos abriéndose
11. ZOOM_OUT - Dos manos cerrándose

### Estructura de archivos:
```
mouXe/
├── mouxe.py              # Programa principal
├── mouxe_ml.py          # Módulo de ML (BiLSTM)
├── recolectar.py        # Recolector de gestos para entrenamiento
├── entrenar.py          # Entrenador del modelo BiLSTM
├── colab_entrenamiento.ipynb  # Notebook para entrenar en Google Colab
├── data/
│   └── gestos_raw.npy   # Datos recolectados
├── models/
│   ├── gesture_model.h5 # Modelo entrenado
│   └── gesture_labels.pkl # Labels del modelo
└── requirements.txt
```

---

## ⚙️ Configuración

Edita las constantes al inicio de [`mouxe.py`](mouxe.py):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `CAMARA` | `0` | Índice de la webcam |
| `NUM_PANTALLAS` | `2` | Número de monitores (ajusta la ganancia del cursor) |
| `SUAVIZADO` | `0.12` | Suavizado del cursor (bajo = suave, alto = ágil) |
| `PINCH_RATIO` | `0.35` | Umbral para detectar pellizco |
| `PINCH_RELEASE` | `0.45` | Umbral para soltar pellizco (histéresis) |
| `FREEZE_RATIO` | `0.55` | Umbral para congelar el cursor (pre-pellizco) |
| `SCROLL_SPEED` | `10` | Velocidad del scroll |
| `MIRROR` | `True` | Espejo horizontal (más natural) |
| `MOSTRAR_CAMARA` | `True` | Mostrar ventana de debug con HUD |

---

## 🔧 Características técnicas

- **Cursor preciso con congelación pre-pellizco**: el cursor se congela automáticamente cuando el pulgar se acerca a cualquier dedo (`FREEZE_RATIO`), evitando desvíos al hacer clic.
- **Detección invariante a escala**: los umbrales de pellizco son relativos al tamaño de la mano (distancia muñeca-nudillo).
- **Histéresis anti-flickeo**: umbrales separados para activar y soltar pellizcos.
- **Soporte multi-monitor**: ajuste automático de ganancia según número de pantallas.
- **Hold y drag**: mantén el pellizco pulgar+índice >0.5s para arrastrar.
- **Botones laterales**: avanzar/retroceder via Win32 API (`ctypes`).
- **Resolución de conflictos**: si varios dedos pellizcan a la vez, gana el más cercano al pulgar.
- **Suavizado del cursor**: interpolación lineal configurable.
- **HUD en tiempo real**: muestra el gesto activo sobre el feed de la cámara.

---

## 📦 Dependencias

```
opencv-python
mediapipe==0.10.9
pyautogui
tensorflow
numpy
```

---

## 📝 Licencia

MIT

---

## 🤝 Créditos

- [MediaPipe](https://google.github.io/mediapipe/) por la detección de manos
- [BiLSTM Gesture Recognition](https://github.com/gspagare/Real-Time-Gesture-Recognition-Using-Mediapipe-and-BiLSTM) por la arquitectura del modelo
