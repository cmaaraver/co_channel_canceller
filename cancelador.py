#!/usr/bin/env python3
"""
SDR Co-Channel Canceller
Aplicación completa para cancelación de interferencias AM/FM
Basado en el plugin original de SDR#
"""

import sys
import os
import numpy as np
import threading
import time
from datetime import datetime

# Qt imports
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, QSlider, QLabel,
    QPushButton, QFileDialog, QComboBox, QCheckBox, QTabWidget, QGroupBox,
    QGridLayout, QMessageBox, QTextEdit, QSpinBox, QRadioButton
)

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from PyQt6.QtGui import QIcon, QFont

# Establecer atributo DPI antes de crear QApplication
try:
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
except AttributeError:
    pass

# matplotlib + sonido + DSP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq

class ModernSlider(QWidget):
    """Slider moderno con display de valor"""
    valueChanged = pyqtSignal(float)
    
    def __init__(self, min_val=0, max_val=100, default_val=50, decimals=1, suffix="", parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.suffix = suffix

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(int((default_val - min_val) / (max_val - min_val) * 1000))

        
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(80)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Estilo moderno
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.3), stop:1 rgba(255,255,255,0.1));
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #007AFF, stop:1 #0051D2);
                border: 2px solid rgba(255,255,255,0.8);
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 11px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0088FF, stop:1 #0061E2);
                border: 2px solid white;
            }
            QLabel {
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 8px;
                padding: 5px;
                color: white;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        
        self.slider.valueChanged.connect(self._on_value_changed)
        self._update_display()
    
    def _on_value_changed(self):
        self._update_display()
        self.valueChanged.emit(self.value())
    
    def _update_display(self):
        val = self.value()
        self.value_label.setText(f"{val:.{self.decimals}f}{self.suffix}")
    
    def value(self):
        slider_val = self.slider.value() / 1000.0
        return self.min_val + slider_val * (self.max_val - self.min_val)
    
    def setValue(self, value):
        slider_val = int((value - self.min_val) / (self.max_val - self.min_val) * 1000)
        self.slider.setValue(slider_val)

class VisualizationWidget(QWidget):
    """Widget de visualización con dominio tiempo/frecuencia/cascada"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.mode = "time"  # time, frequency, waterfall
        self.data_original = np.array([])
        self.data_cancelled = np.array([])
        self.sample_rate = 48000
        self.waterfall_data = []
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controles de visualización
        controls = QHBoxLayout()
        
        self.time_btn = QPushButton("Tiempo")
        self.freq_btn = QPushButton("Frecuencia")
        self.waterfall_btn = QPushButton("Cascada")
        
        for btn in [self.time_btn, self.freq_btn, self.waterfall_btn]:
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(255,255,255,0.15);
                    border: 1px solid rgba(255,255,255,0.3);
                    border-radius: 8px;
                    padding: 8px 16px;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #007AFF, stop:1 #0051D2);
                    border: 1px solid rgba(255,255,255,0.5);
                }
                QPushButton:hover {
                    border: 1px solid rgba(255,255,255,0.5);
                }
            """)
            controls.addWidget(btn)
        
        self.time_btn.setChecked(True)
        controls.addStretch()
        
        self.time_btn.clicked.connect(lambda: self.set_mode("time"))
        self.freq_btn.clicked.connect(lambda: self.set_mode("frequency"))
        self.waterfall_btn.clicked.connect(lambda: self.set_mode("waterfall"))
        
        layout.addLayout(controls)
        
        # Canvas de matplotlib
        self.figure = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: rgba(255,255,255,0.05); border-radius: 12px;")
        layout.addWidget(self.canvas)
        
        # Configurar subplot
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        self.ax1.set_title("Señal Original", color='white', fontweight='bold')
        self.ax2.set_title("Señal Cancelada", color='white', fontweight='bold')
        
    def set_mode(self, mode):
        self.mode = mode
        for btn in [self.time_btn, self.freq_btn, self.waterfall_btn]:
            btn.setChecked(False)
        
        if mode == "time":
            self.time_btn.setChecked(True)
        elif mode == "frequency":
            self.freq_btn.setChecked(True)
        elif mode == "waterfall":
            self.waterfall_btn.setChecked(True)
        
        self.update_plot()
    
    def update_data(self, original, cancelled, sample_rate=48000):
        self.data_original = original
        self.data_cancelled = cancelled
        self.sample_rate = sample_rate
        
        # Para cascada, mantener histórico
        if self.mode == "waterfall":
            self.waterfall_data.append(original)
            if len(self.waterfall_data) > 100:
                self.waterfall_data.pop(0)
        
        self.update_plot()
    
    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        
        if len(self.data_original) == 0:
            self.canvas.draw()
            return
        
        if self.mode == "time":
            self.plot_time_domain()
        elif self.mode == "frequency":
            self.plot_frequency_domain()
        elif self.mode == "waterfall":
            self.plot_waterfall()
        
        # Restaurar estilos
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_time_domain(self):
        t = np.arange(len(self.data_original)) / self.sample_rate
        
        self.ax1.plot(t, self.data_original, color='#FF6B6B', linewidth=1, alpha=0.8)
        self.ax1.set_title("Señal Original", color='white', fontweight='bold')
        self.ax1.set_xlabel("Tiempo (s)", color='white')
        self.ax1.set_ylabel("Amplitud", color='white')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.plot(t, self.data_cancelled, color='#4ECDC4', linewidth=1, alpha=0.8)
        self.ax2.set_title("Señal Cancelada", color='white', fontweight='bold')
        self.ax2.set_xlabel("Tiempo (s)", color='white')
        self.ax2.set_ylabel("Amplitud", color='white')
        self.ax2.grid(True, alpha=0.3)
    
    def plot_frequency_domain(self):
        # FFT de la señal original
        fft_orig = np.abs(fft(self.data_original))
        fft_canc = np.abs(fft(self.data_cancelled))
        freqs = fftfreq(len(self.data_original), 1/self.sample_rate)
        
        # Solo frecuencias positivas
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_orig = fft_orig[pos_mask]
        fft_canc = fft_canc[pos_mask]
        
        self.ax1.semilogy(freqs, fft_orig, color='#FF6B6B', linewidth=1)
        self.ax1.set_title("Espectro Original", color='white', fontweight='bold')
        self.ax1.set_xlabel("Frecuencia (Hz)", color='white')
        self.ax1.set_ylabel("Magnitud", color='white')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.semilogy(freqs, fft_canc, color='#4ECDC4', linewidth=1)
        self.ax2.set_title("Espectro Cancelado", color='white', fontweight='bold')
        self.ax2.set_xlabel("Frecuencia (Hz)", color='white')
        self.ax2.set_ylabel("Magnitud", color='white')
        self.ax2.grid(True, alpha=0.3)
    
    def plot_waterfall(self):
        if len(self.waterfall_data) < 2:
            return
        
        # Crear espectrograma
        waterfall_matrix = np.array(self.waterfall_data).T
        
        extent = [0, len(self.waterfall_data), 0, self.sample_rate/2]
        
        im1 = self.ax1.imshow(np.abs(fft(waterfall_matrix, axis=0))[:len(waterfall_matrix)//2], 
                             aspect='auto', origin='lower', extent=extent, 
                             cmap='plasma', alpha=0.8)
        self.ax1.set_title("Cascada Original", color='white', fontweight='bold')
        self.ax1.set_xlabel("Tiempo", color='white')
        self.ax1.set_ylabel("Frecuencia (Hz)", color='white')
        
        # Para cancelada, usar los últimos datos
        if len(self.data_cancelled) > 0:
            cancelled_fft = np.abs(fft(self.data_cancelled))[:len(self.data_cancelled)//2]
            cancelled_matrix = np.tile(cancelled_fft.reshape(-1, 1), (1, len(self.waterfall_data)))
            
            im2 = self.ax2.imshow(cancelled_matrix, aspect='auto', origin='lower', 
                                 extent=extent, cmap='viridis', alpha=0.8)
            self.ax2.set_title("Cascada Cancelada", color='white', fontweight='bold')
            self.ax2.set_xlabel("Tiempo", color='white')
            self.ax2.set_ylabel("Frecuencia (Hz)", color='white')

class AMCanceller:
    """Cancelador AM basado en el algoritmo original"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.delay = 0.001  # 1ms delay
        self.gain = 0.5
        self.lpf_cutoff = 10000  # 10kHz
        
        # Filtro paso bajo para AM
        self.lpf_b, self.lpf_a = signal.butter(6, self.lpf_cutoff / (sample_rate/2), 'low')
        self.lpf_zi = signal.lfilter_zi(self.lpf_b, self.lpf_a)
        
        # Buffer circular para delay
        self.delay_samples = int(self.delay * sample_rate)
        self.delay_buffer = np.zeros(self.delay_samples)
        self.buffer_index = 0
    
    def process(self, audio_data):
        """Procesa el audio AM con cancelación"""
        if len(audio_data) == 0:
            return audio_data
        
        # Filtro paso bajo
        filtered, self.lpf_zi = signal.lfilter(self.lpf_b, self.lpf_a, audio_data, zi=self.lpf_zi)
        
        # Aplicar delay y cancelación
        delayed_signal = np.zeros_like(filtered)
        
        for i, sample in enumerate(filtered):
            # Obtener muestra con delay
            delayed_sample = self.delay_buffer[self.buffer_index]
            delayed_signal[i] = delayed_sample
            
            # Actualizar buffer circular
            self.delay_buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % self.delay_samples
        
        # Cancelación adaptativa
        cancelled = audio_data - self.gain * delayed_signal
        
        return cancelled
    
    def set_parameters(self, delay, gain, bandwidth):
        self.delay = delay / 1000.0  # ms to seconds
        self.gain = gain
        self.lpf_cutoff = bandwidth
        
        # Recalcular filtro
        self.lpf_b, self.lpf_a = signal.butter(6, self.lpf_cutoff / (self.sample_rate/2), 'low')
        
        # Redimensionar buffer de delay
        new_delay_samples = int(self.delay * self.sample_rate)
        if new_delay_samples != self.delay_samples:
            self.delay_samples = new_delay_samples
            self.delay_buffer = np.zeros(self.delay_samples)
            self.buffer_index = 0

class FMCanceller:
    """Cancelador FM basado en el algoritmo original"""
    
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.deviation = 75000  # 75kHz deviation para FM
        self.delay = 0.0005  # 0.5ms delay
        self.gain = 0.7
        self.hpf_cutoff = 30  # 30Hz high-pass
        
        # Filtro paso alto para FM
        self.hpf_b, self.hpf_a = signal.butter(4, self.hpf_cutoff / (sample_rate/2), 'high')
        self.hpf_zi = signal.lfilter_zi(self.hpf_b, self.hpf_a)
        
        # Pre-énfasis/De-énfasis
        self.preemph_a = np.exp(-1.0 / (sample_rate * 75e-6))
        self.preemph_state = 0
        
        # Buffer circular para delay
        self.delay_samples = int(self.delay * sample_rate)
        self.delay_buffer = np.zeros(self.delay_samples)
        self.buffer_index = 0
    
    def process(self, audio_data):
        """Procesa el audio FM con cancelación"""
        if len(audio_data) == 0:
            return audio_data
        
        # Pre-énfasis
        preemphasized = np.zeros_like(audio_data)
        for i, sample in enumerate(audio_data):
            preemphasized[i] = sample - self.preemph_a * self.preemph_state
            self.preemph_state = sample
        
        # Filtro paso alto
        filtered, self.hpf_zi = signal.lfilter(self.hpf_b, self.hpf_a, preemphasized, zi=self.hpf_zi)
        
        # Aplicar delay y cancelación
        delayed_signal = np.zeros_like(filtered)
        
        for i, sample in enumerate(filtered):
            delayed_sample = self.delay_buffer[self.buffer_index]
            delayed_signal[i] = delayed_sample
            
            self.delay_buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % self.delay_samples
        
        # Cancelación con ganancia adaptativa
        cancelled = audio_data - self.gain * delayed_signal
        
        return cancelled
    
    def set_parameters(self, delay, gain, deviation):
        self.delay = delay / 1000.0
        self.gain = gain
        self.deviation = deviation
        
        # Redimensionar buffer de delay
        new_delay_samples = int(self.delay * self.sample_rate)
        if new_delay_samples != self.delay_samples:
            self.delay_samples = new_delay_samples
            self.delay_buffer = np.zeros(self.delay_samples)
            self.buffer_index = 0

class AudioProcessor(QObject):
    """Procesador principal de audio con threading"""
    
    data_ready = pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self):
        super().__init__()
        self.audio_data = np.array([])
        self.sample_rate = 48000
        self.is_playing = False
        self.is_muted = False
        self.mode = "AM"  # AM or FM
        self.center_freq = 1000000  # 1MHz default
        
        self.am_canceller = AMCanceller(self.sample_rate)
        self.fm_canceller = FMCanceller(self.sample_rate)
        
        self.processing_thread = None
        self.chunk_size = 1024
    
    def load_audio(self, file_path):
        """Cargar archivo de audio"""
        try:
            self.audio_data, self.sample_rate = sf.read(file_path)
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data[:, 0]  # Mono
            
            # Actualizar sample rate en canceladores
            self.am_canceller.sample_rate = self.sample_rate
            self.fm_canceller.sample_rate = self.sample_rate
            
            return True
        except Exception as e:
            print(f"Error cargando audio: {e}")
            return False
    
    def start_processing(self):
        """Iniciar procesamiento en tiempo real"""
        if len(self.audio_data) == 0:
            return
        
        self.is_playing = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Detener procesamiento"""
        self.is_playing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _process_loop(self):
        """Loop principal de procesamiento"""
        position = 0
        
        while self.is_playing and position < len(self.audio_data):
            # Obtener chunk de audio
            end_pos = min(position + self.chunk_size, len(self.audio_data))
            chunk = self.audio_data[position:end_pos]
            
            if len(chunk) == 0:
                position = 0  # Loop
                continue
            
            # Procesar según modo
            if self.mode == "AM":
                processed_chunk = self.am_canceller.process(chunk)
            else:
                processed_chunk = self.fm_canceller.process(chunk)
            
            # Emitir datos procesados
            if not self.is_muted:
                self.data_ready.emit(chunk, processed_chunk, self.sample_rate)
            
            position += self.chunk_size
            
            # Loop infinito
            if position >= len(self.audio_data):
                position = 0
            
            # Control de velocidad de reproducción
            time.sleep(self.chunk_size / self.sample_rate * 0.1)  # 10% real-time
    
    def set_mode(self, mode):
        self.mode = mode
    
    def set_mute(self, muted):
        self.is_muted = muted
    
    def set_am_parameters(self, delay, gain, bandwidth):
        self.am_canceller.set_parameters(delay, gain, bandwidth)
    
    def set_fm_parameters(self, delay, gain, deviation):
        self.fm_canceller.set_parameters(delay, gain, deviation)
    
    def export_audio(self, file_path):
        """Exportar audio procesado"""
        if len(self.audio_data) == 0:
            return False
        
        try:
            # Procesar todo el audio
            if self.mode == "AM":
                processed_audio = self.am_canceller.process(self.audio_data)
            else:
                processed_audio = self.fm_canceller.process(self.audio_data)
            
            # Guardar archivo
            sf.write(file_path, processed_audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error exportando audio: {e}")
            return False

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR Co-Channel Canceller v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Procesador de audio
        self.audio_processor = AudioProcessor()
        self.audio_processor.data_ready.connect(self.update_visualization)
        
        self.setup_ui()
        self.setup_styles()
        self.connect_signals()
        
        # Variables de estado
        self.current_file = ""
        self.is_playing = False
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Panel de controles (izquierda)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 0)
        
        # Panel de visualización (derecha)
        self.visualization = VisualizationWidget()
        main_layout.addWidget(self.visualization, 1)
    
    def create_control_panel(self):
        """Crear panel de controles"""
        panel = QWidget()
        panel.setFixedWidth(350)
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel("Co-Channel Canceller")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: white;
                padding: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #007AFF, stop:1 #0051D2);
                border-radius: 12px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title)
        
        # Controles de archivo
        file_group = self.create_file_controls()
        layout.addWidget(file_group)
        
        # Controles de modo (AM/FM)
        mode_group = self.create_mode_controls()
        layout.addWidget(mode_group)
        
        # Controles de frecuencia
        freq_group = self.create_frequency_controls()
        layout.addWidget(freq_group)
        
        # Controles de cancelación
        cancel_group = self.create_cancellation_controls()
        layout.addWidget(cancel_group)
        
        # Controles de reproducción
        playback_group = self.create_playback_controls()
        layout.addWidget(playback_group)
        
        # Botón de exportar
        export_group = self.create_export_controls()
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return panel
    
    def create_file_controls(self):
        """Crear controles de archivo"""
        group = QGroupBox("Archivo de Audio")
        layout = QVBoxLayout(group)
        
        self.file_label = QLabel("Ningún archivo cargado")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        self.load_btn = QPushButton("Cargar Archivo")
        self.load_btn.clicked.connect(self.load_audio_file)
        layout.addWidget(self.load_btn)
        
        return group
    
    def create_mode_controls(self):
        """Crear controles de modo AM/FM"""
        group = QGroupBox("Modo de Operación")
        layout = QVBoxLayout(group)
        
        mode_layout = QHBoxLayout()
        
        self.am_radio = QRadioButton("AM")
        self.fm_radio = QRadioButton("FM")
        self.am_radio.setChecked(True)
        
        self.am_radio.toggled.connect(self.mode_changed)
        self.fm_radio.toggled.connect(self.mode_changed)
        
        mode_layout.addWidget(self.am_radio)
        mode_layout.addWidget(self.fm_radio)
        
        layout.addLayout(mode_layout)
        
        return group
    
    def create_frequency_controls(self):
        """Crear controles de frecuencia"""
        group = QGroupBox("Configuración de Frecuencia")
        layout = QVBoxLayout(group)
        
        # Frecuencia central
        layout.addWidget(QLabel("Frecuencia Central:"))
        self.center_freq_slider = ModernSlider(87.5, 108.0, 100.0, 1, " MHz")
        self.center_freq_slider.valueChanged.connect(self.center_freq_changed)
        layout.addWidget(self.center_freq_slider)
        
        # Sample Rate
        layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_slider = ModernSlider(8000, 192000, 48000, 0, " Hz")
        self.sample_rate_slider.valueChanged.connect(self.sample_rate_changed)
        layout.addWidget(self.sample_rate_slider)
        
        return group
    
    def create_cancellation_controls(self):
        """Crear controles de cancelación"""
        group = QGroupBox("Parámetros de Cancelación")
        layout = QVBoxLayout(group)
        
        # Delay
        layout.addWidget(QLabel("Delay:"))
        self.delay_slider = ModernSlider(0.1, 10.0, 1.0, 1, " ms")
        self.delay_slider.valueChanged.connect(self.cancellation_params_changed)
        layout.addWidget(self.delay_slider)
        
        # Gain
        layout.addWidget(QLabel("Gain:"))
        self.gain_slider = ModernSlider(0.0, 2.0, 0.5, 2, "")
        self.gain_slider.valueChanged.connect(self.cancellation_params_changed)
        layout.addWidget(self.gain_slider)
        
        # Parámetro específico del modo
        self.mode_param_label = QLabel("Ancho de Banda:")
        layout.addWidget(self.mode_param_label)
        self.mode_param_slider = ModernSlider(1000, 20000, 10000, 0, " Hz")
        self.mode_param_slider.valueChanged.connect(self.cancellation_params_changed)
        layout.addWidget(self.mode_param_slider)
        
        return group
    
    def create_playback_controls(self):
        """Crear controles de reproducción"""
        group = QGroupBox("Control de Reproducción")
        layout = QVBoxLayout(group)
        
        buttons_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.stop_btn = QPushButton("⏹ Stop")
        self.mute_btn = QPushButton("🔊 Mute")
        self.mute_btn.setCheckable(True)
        
        self.play_btn.clicked.connect(self.toggle_playback)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.mute_btn.toggled.connect(self.toggle_mute)
        
        buttons_layout.addWidget(self.play_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.mute_btn)
        
        layout.addLayout(buttons_layout)
        
        return group
    
    def create_export_controls(self):
        """Crear controles de exportación"""
        group = QGroupBox("Exportar Audio")
        layout = QVBoxLayout(group)
        
        self.export_btn = QPushButton("💾 Exportar Audio Procesado")
        self.export_btn.clicked.connect(self.export_audio)
        layout.addWidget(self.export_btn)
        
        return group
    
    def setup_styles(self):
        """Configurar estilos modernos tipo iOS"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2d2d30);
            }
            
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 12px;
                margin-top: 1ex;
                padding-top: 15px;
                background: rgba(255,255,255,0.08);
                backdrop-filter: blur(10px);
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: white;
                font-size: 14px;
            }
            
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.15), stop:1 rgba(255,255,255,0.05));
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 8px;
                padding: 12px 20px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.25), stop:1 rgba(255,255,255,0.15));
                border: 1px solid rgba(255,255,255,0.5);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.05), stop:1 rgba(255,255,255,0.25));
            }
            
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #007AFF, stop:1 #0051D2);
                border: 1px solid rgba(255,255,255,0.5);
            }
            
            QRadioButton {
                color: white;
                font-weight: bold;
                spacing: 10px;
            }
            
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid rgba(255,255,255,0.5);
                background: rgba(255,255,255,0.1);
            }
            
            QRadioButton::indicator:checked {
                background: qradial-gradient(cx:0.5, cy:0.5, radius:0.5,
                    stop:0 #007AFF, stop:0.6 #007AFF, stop:0.7 rgba(255,255,255,0.2));
                border: 2px solid #007AFF;
            }
            
            QLabel {
                color: white;
                font-size: 12px;
                padding: 5px;
            }
        """)
    
    def connect_signals(self):
        """Conectar señales"""
        pass
    
    def load_audio_file(self):
        """Cargar archivo de audio"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Archivo de Audio", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        
        if file_path:
            if self.audio_processor.load_audio(file_path):
                self.current_file = file_path
                filename = os.path.basename(file_path)
                self.file_label.setText(f"Cargado: {filename}")
                
                # Actualizar sample rate slider
                self.sample_rate_slider.setValue(self.audio_processor.sample_rate)
                
                QMessageBox.information(self, "Éxito", "Archivo cargado correctamente")
            else:
                QMessageBox.warning(self, "Error", "No se pudo cargar el archivo")
    
    def mode_changed(self):
        """Cambio de modo AM/FM"""
        if self.am_radio.isChecked():
            self.audio_processor.set_mode("AM")
            self.mode_param_label.setText("Ancho de Banda:")
            self.mode_param_slider.min_val = 1000
            self.mode_param_slider.max_val = 20000
            self.mode_param_slider.setValue(10000)
            self.mode_param_slider.suffix = " Hz"
            
            # Ajustar frecuencia central para AM (530-1700 kHz)
            self.center_freq_slider.min_val = 530
            self.center_freq_slider.max_val = 1700
            self.center_freq_slider.setValue(1000)
            self.center_freq_slider.suffix = " kHz"
            
        else:
            self.audio_processor.set_mode("FM")
            self.mode_param_label.setText("Desviación:")
            self.mode_param_slider.min_val = 25000
            self.mode_param_slider.max_val = 100000
            self.mode_param_slider.setValue(75000)
            self.mode_param_slider.suffix = " Hz"
            
            # Ajustar frecuencia central para FM (87.5-108 MHz)
            self.center_freq_slider.min_val = 87.5
            self.center_freq_slider.max_val = 108.0
            self.center_freq_slider.setValue(100.0)
            self.center_freq_slider.suffix = " MHz"
        
        self.cancellation_params_changed()
    
    def center_freq_changed(self):
        """Cambio de frecuencia central"""
        freq = self.center_freq_slider.value()
        self.audio_processor.center_freq = freq
    
    def sample_rate_changed(self):
        """Cambio de sample rate"""
        sample_rate = int(self.sample_rate_slider.value())
        self.audio_processor.sample_rate = sample_rate
        self.audio_processor.am_canceller.sample_rate = sample_rate
        self.audio_processor.fm_canceller.sample_rate = sample_rate
    
    def cancellation_params_changed(self):
        """Cambio de parámetros de cancelación"""
        delay = self.delay_slider.value()
        gain = self.gain_slider.value()
        mode_param = self.mode_param_slider.value()
        
        if self.am_radio.isChecked():
            self.audio_processor.set_am_parameters(delay, gain, mode_param)
        else:
            self.audio_processor.set_fm_parameters(delay, gain, mode_param)
    
    def toggle_playback(self):
        """Alternar reproducción"""
        if not self.current_file:
            QMessageBox.warning(self, "Advertencia", "Primero carga un archivo de audio")
            return
        
        if not self.is_playing:
            self.audio_processor.start_processing()
            self.play_btn.setText("⏸ Pause")
            self.is_playing = True
        else:
            self.audio_processor.stop_processing()
            self.play_btn.setText("▶ Play")
            self.is_playing = False
    
    def stop_playback(self):
        """Detener reproducción"""
        self.audio_processor.stop_processing()
        self.play_btn.setText("▶ Play")
        self.is_playing = False
    
    def toggle_mute(self, muted):
        """Alternar mute"""
        self.audio_processor.set_mute(muted)
        if muted:
            self.mute_btn.setText("🔇 Unmute")
        else:
            self.mute_btn.setText("🔊 Mute")
    
    def export_audio(self):
        """Exportar audio procesado"""
        if not self.current_file:
            QMessageBox.warning(self, "Advertencia", "Primero carga un archivo de audio")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Audio Procesado", "", 
            "WAV Files (*.wav);;All Files (*)"
        )
        
        if file_path:
            if self.audio_processor.export_audio(file_path):
                QMessageBox.information(self, "Éxito", "Audio exportado correctamente")
            else:
                QMessageBox.warning(self, "Error", "No se pudo exportar el audio")
    
    def update_visualization(self, original, cancelled, sample_rate):
        """Actualizar visualización"""
        self.visualization.update_data(original, cancelled, sample_rate)
    
    def closeEvent(self, event):
        """Evento de cierre"""
        self.audio_processor.stop_processing()
        event.accept()

# Función para crear icono de la aplicación
def create_app_icon():
    """Crear icono de la aplicación como SVG"""
    icon_svg = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#007AFF;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#0051D2;stop-opacity:1" />
            </linearGradient>
        </defs>
        <circle cx="32" cy="32" r="30" fill="url(#grad1)" stroke="white" stroke-width="2"/>
        <path d="M16 32 Q32 16 48 32 Q32 48 16 32" fill="none" stroke="white" stroke-width="3"/>
        <path d="M20 32 Q32 20 44 32 Q32 44 20 32" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="2"/>
        <circle cx="32" cy="32" r="4" fill="white"/>
    </svg>'''
    
    # Crear directorio de iconos si no existe
    icons_dir = os.path.join(os.path.dirname(__file__), "assets", "icons")
    os.makedirs(icons_dir, exist_ok=True)
    
    # Guardar SVG
    svg_path = os.path.join(icons_dir, "signal_icon.svg")
    with open(svg_path, 'w') as f:
        f.write(icon_svg)
    
    return svg_path

def main():
    """Función principal"""
    app = QApplication(sys.argv)
    app.setApplicationName("SDR Co-Channel Canceller")
    app.setApplicationVersion("1.0")
    
    # Crear icono si no existe
    try:
        icon_path = create_app_icon()
        app.setWindowIcon(QIcon(icon_path))
    except:
        pass
    
    # Configurar para alta resolución
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    # Mensaje de bienvenida
    QTimer.singleShot(500, lambda: QMessageBox.information(
        window, "Bienvenido", 
        "SDR Co-Channel Canceller v1.0\n\n"
        "1. Carga un archivo de audio\n"
        "2. Selecciona modo AM o FM\n"
        "3. Ajusta los parámetros\n"
        "4. Presiona Play para ver la cancelación en tiempo real\n"
        "5. Exporta el resultado procesado"
    ))
    
    sys.exit(app.exec())

def main():
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
    
