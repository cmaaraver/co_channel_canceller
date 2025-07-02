# Co-Channel Canceller AM/FM for CleanRF

## Descripción

**Co-Channel Canceller AM/FM for CleanRF** es una herramienta profesional de análisis y procesamiento de señales de radiofrecuencia, diseñada para cancelar interferencias co-canal en señales AM y FM. Inspirada en el funcionamiento del cancelador co-canal de SDR#, permite al usuario ajustar con precisión todos los parámetros relevantes para lograr una cancelación óptima, visualizar el resultado en tiempo real y exportar la señal procesada.

## Funcionalidades

- **Cancelador Co-Canal AM/FM**  
  Permite cancelar interferencias co-canal en señales de tipo AM y FM mediante un algoritmo adaptativo configurable.

- **Controles diferenciados por modo**  
  - **AM:**  
    - Activar/Desactivar cancelador  
    - Eliminar portadora (Remove Carrier)  
    - Auto Tune (ajuste automático de frecuencia de interferencia)  
    - Carrier Offset  
    - Bandwidth  
    - IF Offset  
    - Correction (ajuste de ganancia en dB)
  - **FM:**  
    - Activar/Desactivar cancelador  
    - Carrier Offset  
    - Bandwidth  
    - IF Offset  
    - Sensitivity (ajuste de ganancia en dB)

- **Visualización avanzada**  
  - Dos paneles de visualización independientes: señal original y señal cancelada.
  - Modos de visualización seleccionables: dominio del tiempo, espectro de frecuencia, waterfall.
  - Configuración de parámetros de visualización: tamaño de FFT, tipo de ventana, colores, cuadrícula, autoscale, límites de eje Y, span de frecuencia.
  - Marcadores de offset ajustables y visualización de offset total.

- **Auto Tune**  
  Detección automática de la portadora interferente y ajuste automático del offset para facilitar la cancelación.

- **Exportación de señal**  
  Permite guardar la señal cancelada en formato WAV para su posterior análisis o reproducción.

- **Escaneo de portadoras**  
  Herramienta para detectar y mostrar las portadoras presentes en la señal cargada.

- **Interfaz intuitiva y profesional**  
  - Tooltips explicativos en todos los controles.
  - Botón de reset de zoom para los ejes de visualización.
  - Manejo robusto de errores y mensajes claros al usuario.

## Requisitos

- Python 3.8 o superior
- PyQt6
- numpy
- scipy
- matplotlib
- sounddevice (opcional, para reproducción de audio)
- soundfile (para carga y exportación de archivos WAV)

## Instalación

Instala las dependencias necesarias con pip:

```sh
pip install pyqt6 numpy scipy matplotlib sounddevice soundfile
```

## Uso

1. Ejecuta el script principal:
    ```sh
    python co_channel_canceller_pro.py
    ```
2. Carga un archivo WAV de señal I/Q o audio.
3. Ajusta los parámetros del cancelador según el tipo de señal (AM o FM).
4. Visualiza la señal original y la cancelada en el modo deseado.
5. Utiliza la función Auto Tune para facilitar el ajuste del offset.
6. Exporta la señal cancelada si lo deseas.

## Notas

- El algoritmo de cancelación está inspirado en el funcionamiento del cancelador co-canal de SDR#, utilizando filtrado complejo, ajuste óptimo de ganancia y fase, y resta adaptativa.
- El modo Remove Carrier elimina la portadora de la referencia antes de la cancelación, útil para señales AM.
- El sistema está preparado para señales grabadas; para señales en tiempo real, adapta la entrada según tus necesidades.

## Licencia

Este software se distribuye bajo los términos de la licencia MIT.


