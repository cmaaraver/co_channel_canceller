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

# -------------------------------------------------------------------
# Utilidad: crea un filtro notch de radio r en frecuencia w0 (rad/sample)
# -------------------------------------------------------------------
def make_notch(w0, r=0.98):
    b = [1, -2 * np.cos(w0), 1]
    a = [1, -2 * r * np.cos(w0), r * r]
    return b, a

# -------------------------------------------------------------------
# Canceladores LMS adaptativos con notch para eliminación de portadora
# -------------------------------------------------------------------
class FMCoChannelCanceller:
    def __init__(self, fs):
        self.fs = fs
        self.cancel_on = True

    def update(self, offset, bw, gain, remove_carrier, enabled):
        self.offset = offset
        self.bw = bw
        self.gain = gain
        self.remove_carrier = remove_carrier
        self.cancel_on = enabled
        self.mu = 0.01 * gain * gain
        self.N = max(16, min(512, int(self.fs / self.bw)))
        self.w = np.zeros(self.N, dtype=complex)
        self.buf = np.zeros(self.N, dtype=complex)
        w0 = 2 * np.pi * offset / self.fs
        self.b_notch, self.a_notch = make_notch(w0)

    def process(self, data):
        if len(data) == 0 or not self.cancel_on:
            # Passthrough
            return data.real if np.iscomplexobj(data) else data.copy()

        t = np.arange(len(data)) / self.fs
        ref = np.exp(1j * 2 * np.pi * self.offset * t)
        x = data * ref
        if self.remove_carrier:
            x = sig.lfilter(self.b_notch, self.a_notch, x)

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
        self.cancel_on = True

    def update(self, offset, bw, gain, remove_carrier, enabled):
        self.offset = offset
        self.bw = bw
        self.gain = gain
        self.remove_carrier = remove_carrier
        self.cancel_on = enabled
        self.mu = 0.01 * gain * gain
        self.N = max(16, min(512, int(self.fs / self.bw)))
        self.w = np.zeros(self.N)
        self.buf = np.zeros(self.N)
        w0 = 2 * np.pi * offset / self.fs
        self.b_notch, self.a_notch = make_notch(w0)

    def process(self, data):
        if len(data) == 0 or not self.cancel_on:
            return data.copy()

        t = np.arange(len(data)) / self.fs
        ref = np.cos(2 * np.pi * self.offset * t)
        x = data * ref
        if self.remove_carrier:
            x = sig.lfilter(self.b_notch, self.a_notch, x)

        y = np.zeros_like(data)
        for n in range(len(data)):
            self.buf = np.roll(self.buf, -1)
            self.buf[-1] = x[n]
            est = np.dot(self.w, self.buf)
            err = data[n] - self.gain * est
            self.w += self.mu * err * self.buf
            y[n] = err
        return y

# -------------------------------------------------------------------
# Visualizador profesional: time, freq y waterfall en entorno oscuro
# -------------------------------------------------------------------
class Visualizer(FigureCanvas):
    def __init__(self, title=""):
        fig = Figure(facecolor="black")
        super().__init__(fig)
        self.ax = fig.add_subplot(111, facecolor="black")
        self.title = title
        self.mode = 'time'
        self.fft_size = 1024
        self.window_type = 'hann'  # Cambiado aquí
        self.grid_on = True
        self.axis_on = True
        self.autoscale = True
        self.ymin = None
        self.ymax = None
        self.ref_level = 0.0
        self.hist_len = 100
        self.history = None
        self.line_color = 'cyan'
        self._apply_theme()

    def _apply_theme(self):
        ax = self.ax
        ax.set_facecolor("black")
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    def set_mode(self, mode):
        self.mode = mode

    def set_line_color(self, color):
        self.line_color = color

    def set_settings(self, fft, window, grid, axis, autoscale,
                     ymin, ymax, ref_level):
        self.fft_size = fft
        self.window_type = window  # Cambiado aquí
        self.grid_on = grid
        self.axis_on = axis
        self.autoscale = autoscale
        self.ymin = ymin
        self.ymax = ymax
        self.ref_level = ref_level

    def refresh(self, data, fs, center_hz=0, mark_hz=None):
        if data is None or len(data) == 0:
            return

        self.ax.clear()
        self._apply_theme()

        N = self.fft_size
        d = np.zeros(N, dtype=complex)
        d[:min(len(data), N)] = data[:min(len(data), N)]
        if not np.iscomplexobj(d):
            d = sig.hilbert(d)
        window = sig.get_window(self.window_type, N, fftbins=True)  # Cambiado aquí
        d *= window

        if self.mode == 'time':
            t = np.arange(len(data)) / fs
            self.ax.plot(t * 1e3, np.real(data), color=self.line_color)
            if self.axis_on:
                self.ax.set_xlabel("Tiempo (ms)")
                self.ax.set_ylabel("Amplitud")
        else:
            spec = np.fft.fftshift(np.fft.fft(d))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
            mag = 20 * np.log10(np.abs(spec) + 1e-6)
            axis = freqs / 1e6 + center_hz / 1e6
            if self.mode == 'freq':
                self.ax.plot(axis, mag, color=self.line_color)
            else:  # waterfall
                if self.history is None or self.history.shape[1] != N:
                    self.history = np.zeros((self.hist_len, N))
                self.history = np.roll(self.history, -1, axis=0)
                self.history[-1, :] = mag
                extent = [axis[0], axis[-1], 0, self.hist_len]
                self.ax.imshow(self.history, aspect='auto', cmap='plasma',
                               origin='upper', extent=extent)
            if mark_hz is not None:
                self.ax.axvline(mark_hz / 1e6, color='red', linestyle='--')
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

# -------------------------------------------------------------------
# Ventana principal con scroll y switch estilo iPhone
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clean RF Canceller — Profesional")
        self.resize(1200, 950)

        self.data = np.array([])
        self.fs = 48000
        self.ptr = 0
        self.chunk = 2048

        self.fm = FMCoChannelCanceller(self.fs)
        self.am = AMCoChannelCanceller(self.fs)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)

        self._apply_theme()
        self._build_ui()

    def _apply_theme(self):
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
            /* Switch estilo iPhone */
            QCheckBox::indicator {
                width: 50px; height: 25px;
                border-radius: 12px; background: #666;
            }
            QCheckBox::indicator:checked {
                background: #44c767;
            }
        """)

    def _build_ui(self):
        # Contenedor con scroll
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        layout = QVBoxLayout(container)

        # ---- Top bar ----
        bar = QHBoxLayout()
        self.load_btn = QPushButton("📂 Cargar WAV")
        self.start_btn = QPushButton("▶ Iniciar")
        self.stop_btn = QPushButton("⏹ Detener")
        self.mode_cb = QComboBox(); self.mode_cb.addItems(["FM", "AM"])
        self.cancel_chk = QCheckBox("Cancelación ON"); self.cancel_chk.setChecked(True)
        self.file_label = QLabel("Archivo: (ninguno)")
        self.file_label.setStyleSheet("color:white")
        bar.addWidget(self.load_btn)
        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(QLabel("Modo:"))
        bar.addWidget(self.mode_cb)
        bar.addWidget(self.cancel_chk)
        bar.addWidget(self.file_label)
        bar.addStretch()
        layout.addLayout(bar)

        # ---- Parámetros de señal ----
        params = QHBoxLayout()
        self.center_spin = QDoubleSpinBox(); self.center_spin.setRange(0,6000); self.center_spin.setDecimals(3); self.center_spin.setValue(100.000)
        self.off_spin    = QDoubleSpinBox(); self.off_spin.setRange(0,6000);    self.off_spin.setDecimals(3);    self.off_spin.setValue(0.0)
        self.bw_spin     = QDoubleSpinBox(); self.bw_spin.setRange(100,200000);self.bw_spin.setSingleStep(100); self.bw_spin.setValue(12000)
        self.gain_sld    = QSlider(Qt.Orientation.Horizontal); self.gain_sld.setRange(-60,20); self.gain_sld.setValue(0)
        self.gain_label  = QLabel("0 dB")
        self.gain_sld.valueChanged.connect(lambda v: self.gain_label.setText(f"{v} dB"))
        self.chk_rc      = QCheckBox("Eliminar portadora")
        self.chk_at      = QCheckBox("Auto Tune (AM)")
        self.view_cb     = QComboBox(); self.view_cb.addItems(["time","freq","waterfall"])
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
            if lbl: params.addWidget(QLabel(lbl))
            params.addWidget(w)
        layout.addLayout(params)

        # ---- Panel de visualización ----
        vis_box = QGroupBox("Configuración de visualización")
        vis_layout = QHBoxLayout(vis_box)
        self.chk_grid    = QCheckBox("Grid");    self.chk_grid.setChecked(True)
        self.chk_axis    = QCheckBox("Ejes");    self.chk_axis.setChecked(True)
        self.chk_auto    = QCheckBox("Autoscale");self.chk_auto.setChecked(True)
        self.ymin_spin   = QDoubleSpinBox(); self.ymin_spin.setRange(-200,200); self.ymin_spin.setValue(-100)
        self.ymax_spin   = QDoubleSpinBox(); self.ymax_spin.setRange(-200,200); self.ymax_spin.setValue(0)
        self.ref_spin    = QDoubleSpinBox(); self.ref_spin.setRange(-200,200);    self.ref_spin.setValue(0)
        self.fft_cb      = QComboBox(); self.fft_cb.addItems(["512","1024","2048","4096"])
        self.win_cb      = QComboBox(); self.win_cb.addItems(["rectangular","hann","hamming","blackman"])
        self.color_o_cb  = QComboBox(); self.color_o_cb.addItems(["cyan","lime","red","magenta","yellow","white"])
        self.color_c_cb  = QComboBox(); self.color_c_cb.addItems(["orange","white","red","blue","green","magenta"])
        for w, lbl in [
            (self.chk_grid,    None), (self.chk_axis,   None), (self.chk_auto, None),
            (self.ymin_spin,  "Ymin:"), (self.ymax_spin,  "Ymax:"), (self.ref_spin,"Ref:"),
            (self.fft_cb,     "FFT:"), (self.win_cb,     "Ventana:"),
            (self.color_o_cb, "Color orig.:"),(self.color_c_cb,"Color canc.:")
        ]:
            if lbl: vis_layout.addWidget(QLabel(lbl))
            vis_layout.addWidget(w)
        layout.addWidget(vis_box)

        # ---- Visualizadores ----
        self.vis_o = Visualizer("Original")
        self.vis_c = Visualizer("Cancelada")
        layout.addWidget(self.vis_o)
        layout.addWidget(self.vis_c)

        # ---- Conexiones ----
        self.load_btn.clicked.connect(self.load_wav)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.timer.stop)
        self.view_cb.currentTextChanged.connect(lambda m: (self.vis_o.set_mode(m), self.vis_c.set_mode(m)))
        self.mode_cb.currentTextChanged.connect(lambda _: setattr(self, 'ptr', 0))
        # panel visual
        for w in [self.chk_grid, self.chk_axis, self.chk_auto,
                  self.ymin_spin, self.ymax_spin, self.ref_spin,
                  self.fft_cb, self.win_cb, self.color_o_cb, self.color_c_cb]:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(self.update_view_settings)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self.update_view_settings)
            else:
                w.valueChanged.connect(self.update_view_settings)

    def update_view_settings(self, *_):
        fft = int(self.fft_cb.currentText())
        win = self.win_cb.currentText()
        grid = self.chk_grid.isChecked()
        axis = self.chk_axis.isChecked()
        autos = self.chk_auto.isChecked()
        ymin = self.ymin_spin.value()
        ymax = self.ymax_spin.value()
        ref = self.ref_spin.value()
        col_o = self.color_o_cb.currentText()
        col_c = self.color_c_cb.currentText()
        for vis, col in [(self.vis_o, col_o), (self.vis_c, col_c)]:
            vis.set_settings(fft, win, grid, axis, autos, ymin, ymax, ref)
            vis.set_line_color(col)

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar WAV", filter="WAV files (*.wav)")
        if not path:
            return
        data, fs = sf.read(path)
        if data.ndim > 1 and data.shape[1] >= 2:
            data = data[:,0] + 1j * data[:,1]
        elif data.ndim > 1:
            data = data[:,0]
        if data.size == 0:
            QMessageBox.warning(self, "Error", "El archivo WAV está vacío.")
            return
        self.data = data
        self.fs = fs
        self.ptr = 0
        self.fm = FMCoChannelCanceller(self.fs)
        self.am = AMCoChannelCanceller(self.fs)
        fname = path.split("/")[-1].split("\\")[-1]
        self.file_label.setText(f"Archivo: {fname}")

    def on_start(self):
        if self.data.size == 0:
            QMessageBox.warning(self, "Error", "Carga primero un archivo WAV.")
            return
        self.ptr = 0
        self.timer.start(40)

    def update_loop(self):
        start = self.ptr
        end = start + self.chunk
        if end > len(self.data):
            self.ptr = 0
            start = 0
            end = self.chunk

        seg = self.data[start:end]
        self.ptr += self.chunk

        center_hz = self.center_spin.value() * 1e6
        offset_hz = self.off_spin.value() * 1e6
        bw = self.bw_spin.value()
        gain_db = self.gain_sld.value()
        gain = 10 ** (gain_db / 20)
        rc = self.chk_rc.isChecked()
        enabled = self.cancel_chk.isChecked()

        if self.mode_cb.currentText() == "FM":
            self.fm.update(offset_hz, bw, gain, rc, enabled)
            proc = self.fm.process(seg)
            mark = center_hz + offset_hz
        else:
            self.am.update(offset_hz, bw, gain, rc, enabled)
            proc = self.am.process(seg)
            mark = center_hz + offset_hz

        self.vis_o.refresh(seg, self.fs, center_hz)
        self.vis_c.refresh(proc, self.fs, center_hz, mark)

# -------------------------------------------------------------------
# Lanzamiento
# -------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
