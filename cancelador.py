#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import soundfile as sf
import scipy.signal as sig
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QCheckBox,
    QComboBox, QDoubleSpinBox, QMessageBox, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# === Canceladores LMS Adaptativos ===
class FMCoChannelCanceller:
    def __init__(self, fs):
        self.fs = fs

    def update(self, offset, bw, gain, rc):
        self.offset = offset
        self.bw = bw
        self.gain = gain
        self.rc = rc
        self.mu = 0.01 * gain**2
        self.N = max(16, min(512, int(self.fs / self.bw)))
        self.w = np.zeros(self.N, dtype=complex)
        self.buf = np.zeros(self.N, dtype=complex)

    def process(self, data):
        if len(data) == 0:
            return data
        t = np.arange(len(data)) / self.fs
        ref = np.exp(1j * 2 * np.pi * self.offset * t)
        x = data * ref
        if self.rc:
            x = x - np.mean(x)
        y = np.zeros_like(data, dtype=float)
        for n in range(len(data)):
            self.buf = np.roll(self.buf, -1)
            self.buf[-1] = x[n]
            est = np.dot(np.conj(self.w), self.buf)
            err = data[n] - self.gain * est
            self.w += self.mu * err * self.buf
            y[n] = err.real
        return y

class AMCoChannelCanceller:
    def __init__(self, fs):
        self.fs = fs

    def update(self, offset, bw, gain, rc, _):
        self.offset = offset
        self.bw = bw
        self.gain = gain
        self.rc = rc
        self.mu = 0.01 * gain**2
        self.N = max(16, min(512, int(self.fs / self.bw)))
        self.w = np.zeros(self.N)
        self.buf = np.zeros(self.N)

    def process(self, data):
        if len(data) == 0:
            return data
        t = np.arange(len(data)) / self.fs
        ref = np.cos(2 * np.pi * self.offset * t)
        x = data * ref
        if self.rc:
            x = x - np.mean(x)
        y = np.zeros_like(data)
        for n in range(len(data)):
            self.buf = np.roll(self.buf, -1)
            self.buf[-1] = x[n]
            est = np.dot(self.w, self.buf)
            err = data[n] - self.gain * est
            self.w += self.mu * err * self.buf
            y[n] = err
        return y

# === Visualizador Profesional ===
class Visualizer(FigureCanvas):
    def __init__(self, title=""):
        fig = Figure(facecolor="black")
        super().__init__(fig)
        self.ax = fig.add_subplot(111, facecolor="black")
        self.title = title
        self.mode = 'time'
        self.fft_size = 1024
        self.window_type = 'hann'  
        self.grid_on = True
        self.axis_on = True
        self.autoscale = True
        self.ymin = None
        self.ymax = None
        self.ref_level = 0.0
        self.hist_len = 100
        self.history = None
        self.line_color = 'cyan'
        self._theme()

    def _theme(self):
        self.ax.set_facecolor("black")
        self.ax.title.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.tick_params(colors='white')
        for s in self.ax.spines.values():
            s.set_color('white')

    def set_mode(self, m): self.mode = m
    def set_line_color(self, c): self.line_color = c
    def set_settings(self, fft, win, grid, axis, auto, ymin, ymax, ref):
        self.fft_size = fft
        self.window_type = win 
        self.grid_on = grid
        self.axis_on = axis
        self.autoscale = auto
        self.ymin = ymin
        self.ymax = ymax
        self.ref_level = ref

    def refresh(self, data, fs, center=0, mark=None):
        if data is None or fs is None or len(data) == 0:
            return
        self.ax.clear()
        self._theme()
        N = self.fft_size
        d = np.zeros(N, dtype=complex)
        d[:min(len(data), N)] = data[:min(len(data), N)]
        if not np.iscomplexobj(d):
            d = sig.hilbert(d)
        # Cambiado aquí ↓↓↓
        d *= sig.get_window(self.window_type, N, fftbins=True)
        if self.mode == 'time':
            t = np.arange(len(data)) / fs
            self.ax.plot(t * 1e3, np.real(data), color=self.line_color)
            if self.axis_on:
                self.ax.set_xlabel("Tiempo (ms)")
                self.ax.set_ylabel("Amplitud")
        elif self.mode in ['freq', 'waterfall']:
            spec = np.fft.fftshift(np.fft.fft(d))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
            mag = 20 * np.log10(np.abs(spec) + 1e-6)
            x_axis = freqs / 1e6 + center / 1e6
            if self.mode == 'freq':
                self.ax.plot(x_axis, mag, color=self.line_color)
            else:
                if self.history is None or self.history.shape[1] != N:
                    self.history = np.zeros((self.hist_len, N))
                self.history = np.roll(self.history, -1, axis=0)
                self.history[-1, :] = mag
                extent = [x_axis[0], x_axis[-1], 0, self.hist_len]
                self.ax.imshow(self.history, aspect='auto', cmap='plasma',
                               origin='upper', extent=extent)
            if mark:
                self.ax.axvline(mark / 1e6, color='red', linestyle='--')
            if self.axis_on:
                self.ax.set_xlabel("Frecuencia (MHz)")
                if self.mode == 'freq':
                    self.ax.set_ylabel("Magnitud (dB)")
            self.ax.axhline(self.ref_level, color='white', linestyle=':')
            if not self.autoscale and self.ymin is not None and self.ymax is not None:
                self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.grid(self.grid_on, color='white', alpha=0.2)
        self.ax.set_title(self.title)
        self.draw()
# === Interfaz Profesional ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clean RF Canceller — Profesional")
        self.resize(1200, 950)
        self.data = np.array([])
        self.fs = 48000
        self.ptr = 0
        self.chunk = 2048
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.fm = FMCoChannelCanceller(self.fs)
        self.am = AMCoChannelCanceller(self.fs)
        self._theme()
        self._ui()

    def _theme(self):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#121212"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        self.setPalette(pal)
        self.setStyleSheet("""
            QLabel,QCheckBox,QComboBox,QPushButton,QDoubleSpinBox {
                color:white; background:#2c2c2c;
            }
            QSlider::groove:horizontal { background:#444; height:6px; }
            QSlider::handle:horizontal { background:white; width:10px; margin:-5px 0; }
            QGroupBox { color:white; border:1px solid #444; margin-top:10px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 3px; }
        """)

    def _ui(self):
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
        v = QVBoxLayout(container)

        # Sección Entrada
        h1 = QHBoxLayout()
        self.load_btn = QPushButton("📂 Cargar WAV")
        self.start_btn = QPushButton("▶ Iniciar")
        self.stop_btn = QPushButton("⏹ Detener")
        self.mode_cb = QComboBox(); self.mode_cb.addItems(["FM", "AM"])
        self.file_label = QLabel("Archivo: (ninguno)")
        self.file_label.setStyleSheet("color:white")
        h1.addWidget(self.load_btn)
        h1.addWidget(self.start_btn)
        h1.addWidget(self.stop_btn)
        h1.addWidget(QLabel("Modo:"))
        h1.addWidget(self.mode_cb)
        h1.addWidget(self.file_label)
        h1.addStretch()
        v.addLayout(h1)

        # Sección Parámetros
        h2 = QHBoxLayout()
        self.center_spin = QDoubleSpinBox(); self.center_spin.setRange(0, 6000); self.center_spin.setDecimals(3); self.center_spin.setValue(100.000)
        self.off_spin = QDoubleSpinBox(); self.off_spin.setRange(0, 6000); self.off_spin.setDecimals(3); self.off_spin.setValue(0.0)
        self.bw_spin = QDoubleSpinBox(); self.bw_spin.setRange(100, 200000); self.bw_spin.setSingleStep(100); self.bw_spin.setValue(12000)
        self.gain_sld = QSlider(Qt.Orientation.Horizontal); self.gain_sld.setRange(-60, 20); self.gain_sld.setValue(0)
        self.gain_label = QLabel("0 dB")
        self.gain_sld.valueChanged.connect(lambda v: self.gain_label.setText(f"{v} dB"))
        self.chk_rc = QCheckBox("Eliminar portadora")
        self.chk_at = QCheckBox("Auto Tune (AM)")
        self.view_cb = QComboBox(); self.view_cb.addItems(["time", "freq", "waterfall"])
        for w, lbl in [
            (self.center_spin, "Center (MHz):"),
            (self.off_spin,    "Offset (MHz):"),
            (self.bw_spin,     "BW (Hz):"),
            (self.gain_sld,    "Ganancia (dB):"),
            (self.gain_label,  None),
            (self.chk_rc,      None),
            (self.chk_at,      None),
            (self.view_cb,     "Vista:")
        ]:
            if lbl: h2.addWidget(QLabel(lbl))
            h2.addWidget(w)
        v.addLayout(h2)

        # Panel visual settings
        vis_box = QGroupBox("Configuración de visualización")
        h3 = QHBoxLayout(vis_box)
        self.chk_grid = QCheckBox("Grid"); self.chk_grid.setChecked(True)
        self.chk_axis = QCheckBox("Etiquetas de ejes"); self.chk_axis.setChecked(True)
        self.chk_auto = QCheckBox("Autoscale"); self.chk_auto.setChecked(True)
        self.ymin_spin = QDoubleSpinBox(); self.ymin_spin.setRange(-200, 200); self.ymin_spin.setValue(-100)
        self.ymax_spin = QDoubleSpinBox(); self.ymax_spin.setRange(-200, 200); self.ymax_spin.setValue(0)
        self.ref_spin  = QDoubleSpinBox(); self.ref_spin.setRange(-200, 200); self.ref_spin.setValue(0)
        self.fft_cb    = QComboBox(); self.fft_cb.addItems(["512", "1024", "2048", "4096"])
        self.win_cb    = QComboBox(); self.win_cb.addItems(["rectangular", "hann", "hamming", "blackman"])
        self.color_o_cb= QComboBox(); self.color_o_cb.addItems(["cyan", "lime", "red", "magenta", "yellow", "white"])
        self.color_c_cb= QComboBox(); self.color_c_cb.addItems(["orange", "white", "red", "blue", "green", "magenta"])
        for w, lbl in [
            (self.chk_grid,   None), (self.chk_axis,   None), (self.chk_auto,   None),
            (self.ymin_spin,  "Ymin:"), (self.ymax_spin,  "Ymax:"), (self.ref_spin, "Ref:"),
            (self.fft_cb,     "FFT:"), (self.win_cb,     "Ventana:"),
            (self.color_o_cb, "Color original:"), (self.color_c_cb, "Color cancelada:")
        ]:
            if lbl: h3.addWidget(QLabel(lbl))
            h3.addWidget(w)
        v.addWidget(vis_box)

        # Visualizadores
        self.vis_o = Visualizer("Original")
        self.vis_c = Visualizer("Cancelada")
        v.addWidget(self.vis_o)
        v.addWidget(self.vis_c)

        # Conexiones
        self.load_btn.clicked.connect(self.load_wav)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.timer.stop)
        self.mode_cb.currentTextChanged.connect(lambda _: setattr(self, 'ptr', 0))
        self.view_cb.currentTextChanged.connect(lambda m: (
            self.vis_o.set_mode(m), self.vis_c.set_mode(m)
        ))
        for w in [
            self.chk_grid, self.chk_axis, self.chk_auto,
            self.ymin_spin, self.ymax_spin, self.ref_spin,
            self.fft_cb, self.win_cb,
            self.color_o_cb, self.color_c_cb
        ]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(self.update_view_settings)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self.update_view_settings)
            else:
                w.valueChanged.connect(self.update_view_settings)

    def update_view_settings(self, *_):
        fft_size = int(self.fft_cb.currentText())
        window   = self.win_cb.currentText()
        grid     = self.chk_grid.isChecked()
        axis     = self.chk_axis.isChecked()
        autos    = self.chk_auto.isChecked()
        ymin     = self.ymin_spin.value()
        ymax     = self.ymax_spin.value()
        ref      = self.ref_spin.value()
        col_o    = self.color_o_cb.currentText()
        col_c    = self.color_c_cb.currentText()
        for vis, col in [(self.vis_o, col_o), (self.vis_c, col_c)]:
            vis.set_settings(fft_size, window, grid, axis, autos, ymin, ymax, ref)
            vis.set_line_color(col)

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar WAV", filter="WAV files (*.wav)")
        if not path: return
        data, fs = sf.read(path)
        if data.ndim > 1 and data.shape[1] >= 2:
            data = data[:, 0] + 1j * data[:, 1]
        elif data.ndim > 1:
            data = data[:, 0]
        if data.size == 0:
            QMessageBox.warning(self, "Error", "WAV sin datos.")
            return
        self.data, self.fs, self.ptr = data, fs, 0
        self.fm = FMCoChannelCanceller(fs)
        self.am = AMCoChannelCanceller(fs)
        fname = path.split("/")[-1].split("\\")[-1]
        self.file_label.setText(f"Archivo: " + fname)

    def on_start(self):
        if self.data.size == 0:
            QMessageBox.warning(self, "Error", "Carga primero un archivo WAV.")
            return
        self.ptr = 0
        self.timer.start(40)  # ~25 fps

    def update_loop(self):
        start = self.ptr
        end = start + self.chunk
        if end > len(self.data):
            self.ptr = 0
            start = 0
            end = self.chunk

        seg = self.data[start:end]
        self.ptr += self.chunk

        # lectura de parámetros
        center_hz = self.center_spin.value() * 1e6
        offset_hz = self.off_spin.value() * 1e6
        bw = self.bw_spin.value()
        gain_db = self.gain_sld.value()
        gain = 10 ** (gain_db / 20)
        rc = self.chk_rc.isChecked()
        at = self.chk_at.isChecked()

        if self.mode_cb.currentText() == "FM":
            self.fm.update(offset_hz, bw, gain, rc)
            proc = self.fm.process(seg)
            mark = center_hz + offset_hz
        else:
            self.am.update(offset_hz, bw, gain, rc, at)
            proc = self.am.process(seg)
            mark = center_hz + offset_hz

        self.vis_o.refresh(seg, self.fs, center_hz)
        self.vis_c.refresh(proc, self.fs, center_hz, mark)

# === Lanzamiento de la aplicación ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
