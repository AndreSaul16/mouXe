# MouXe

Controla tu ordenador con una sola mano usando la webcam. Sin hardware extra, sin sensores, solo tu mano y una camara.

MouXe usa **MediaPipe** para detectar la mano en tiempo real y traduce gestos naturales en acciones del raton: mover el cursor, clic izquierdo/derecho, scroll, arrastrar, avanzar y retroceder en el navegador.

---

## Controles

Funciona igual con mano izquierda o derecha.

| Gesto | Accion |
|-------|--------|
| ☝️ Indice levantado | Mover cursor |
| 🤏 Pulgar + indice | Clic izquierdo (sostener >0.5s = hold/arrastrar) |
| 🖕 Pulgar + corazon | Clic derecho |
| 💍 Pulgar + anular | Avanzar (boton lateral del mouse) |
| 🤙 Pulgar + menique | Retroceder (boton lateral del mouse) |
| ✌️ Indice + corazon levantados | Scroll (mover arriba/abajo) |
| 🤌 Todos juntos (marioneta) | Scroll hold (boton central mantenido) |

---

## Como funciona

```
Webcam → MediaPipe Hands (21 landmarks) → Motor de gestos → Input del SO (raton)
```

### Caracteristicas clave

- **Cursor preciso con congelacion pre-pellizco**: el cursor se congela automaticamente cuando el pulgar se acerca a cualquier dedo (`FREEZE_RATIO`), evitando desvios al hacer clic. Apuntas con el indice, congelas, y haces clic sin que el cursor se mueva.
- **Deteccion invariante a escala**: los umbrales de pellizco son relativos al tamano de la mano (distancia muneca-nudillo), funciona igual a 40cm o 80cm de la camara.
- **Histeresis anti-flickeo**: umbrales separados para activar (`PINCH_RATIO=0.35`) y soltar (`PINCH_RELEASE=0.45`) pellizcos, creando una zona muerta que evita disparos falsos.
- **Soporte multi-monitor**: ajuste automatico de ganancia segun numero de pantallas (`NUM_PANTALLAS`).
- **Hold y drag**: manten el pellizco pulgar+indice >0.5s para arrastrar. El cursor sigue el punto medio entre pulgar e indice.
- **Botones laterales**: avanzar/retroceder via Win32 API (`ctypes`), sin dependencias extra.
- **Resolucion de conflictos**: si varios dedos pellizcan a la vez, gana el mas cercano al pulgar.
- **Suavizado del cursor**: interpolacion lineal configurable (`SUAVIZADO`) para evitar temblores.
- **HUD en tiempo real**: muestra el gesto activo sobre el feed de la camara.

---

## Requisitos

- Python 3.8+
- Windows (los botones avanzar/retroceder usan Win32 API)
- Webcam

> **Nota:** Requiere `mediapipe==0.10.9`. Versiones 0.10.10+ eliminaron la API `mp.solutions` y no son compatibles.

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

```bash
python mouxe.py
```

Presiona **Q** en la ventana de la camara para salir. Todos los botones mantenidos se liberan automaticamente al cerrar.

---

## Configuracion

Edita las constantes al inicio de `mouxe.py`:

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `CAMARA` | `0` | Indice de la webcam |
| `NUM_PANTALLAS` | `2` | Numero de monitores (ajusta la ganancia del cursor) |
| `SUAVIZADO` | `0.12` | Suavizado del cursor (bajo = suave, alto = agil) |
| `PINCH_RATIO` | `0.35` | Umbral para detectar pellizco |
| `PINCH_RELEASE` | `0.45` | Umbral para soltar pellizco (histeresis) |
| `FREEZE_RATIO` | `0.55` | Umbral para congelar el cursor (pre-pellizco) |
| `SCROLL_SPEED` | `10` | Velocidad del scroll |
| `MIRROR` | `True` | Espejo horizontal (mas natural) |
| `MOSTRAR_CAMARA` | `True` | Mostrar ventana de debug con HUD |

---

## Dependencias

```
opencv-python
mediapipe==0.10.9
pyautogui
```

---

## Licencia

MIT
