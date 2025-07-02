import sys
import numpy as np
import scipy.signal as sig
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QSlider, QDoubleSpinBox, QComboBox, QFileDialog, QGroupBox, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import sounddevice as sd
except ImportError:
    sd = None

# === Cancelador único AM/FM ===

class CoChannelCanceller:
    def __init__(self, fs):
        self.fs = fs
        self.offset = 0
        self.bw = 12000.0
        self.gain = 1.0
        self.enabled = False
        self.remove_carrier = False
        self.mode = "AM"
        self._need_config = True
        self._last_offset = None
        self._last_bw = None
        self._last_fs = None
        self._filter = None
        self._filter_delay = 0

    def reset(self):
        self._need_config = True

    def update(self, offset, bw, gain, remove_carrier, enabled, mode):
        self.offset = offset
        self.bw = bw
        self.gain = gain
        self.remove_carrier = remove_carrier
        self.enabled = enabled
        self.mode = mode
        self._need_config = True

    def _configure(self):
        N = 255 if self.mode == "AM" else 63
        f0 = self.offset / self.fs
        taps = sig.firwin(N, self.bw / self.fs, window='hann')
        t = np.arange(N)
        osc = np.exp(-1j * 2 * np.pi * f0 * t)
        kernel = taps * osc
        self._filter = kernel
        self._filter_delay = N // 2
        self._last_offset = self.offset
        self._last_bw = self.bw
        self._last_fs = self.fs
        self._need_config = False

    def process(self, x):
        if not self.enabled or len(x) == 0:
            return x.copy()
        if self._need_config or self._last_offset != self.offset or self._last_bw != self.bw or self._last_fs != self.fs:
            self._configure()
        # Filtro pasa banda centrado en offset
        x_filt = sig.lfilter(self._filter, [1.0], x)
        x_filt = np.roll(x_filt, -self._filter_delay)
        # Ajuste de ganancia y fase óptima (como SDR#)
        alpha = np.vdot(x, x_filt) / (np.vdot(x_filt, x_filt) + 1e-12)
        ref = alpha * x_filt
        # Remove carrier (DC) si se pide
        if self.remove_carrier:
            ref = ref - np.mean(ref)
            if np.allclose(ref, 0, atol=1e-6):
                return np.zeros_like(x)
        # Resta la referencia adaptada
        y = x - self.gain * ref
        # Normalización para evitar amplificación de ruido
        if np.std(y) > 2 * np.std(x):
            y = y * (np.std(x) / (np.std(y) + 1e-6))
        return y

def demod_fm(x):
    if not np.iscomplexobj(x):
        x = sig.hilbert(x)
    return np.diff(np.unwrap(np.angle(x)))

def demod_am(x):
    return np.abs(x)

class Visualizer(FigureCanvas):
    def __init__(self, title=""):
        fig = Figure(figsize=(8, 4), facecolor="black")
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.title = title
        self.window_type = 'hann'
        self.line_color = 'cyan'
        self.fft_size = 2048
        self.span_mhz = 2.0
        self.auto_scale = True
        self.ymin_db = -100
        self.ymax_db = 0
        self.grid_on = True
        self.mode = "Frecuencia"
        self.offsets = []

    def refresh(self, data, fs, center_hz=0, marks=[]):
        self.ax.clear()
        N = self.fft_size
        if not np.iscomplexobj(data):
            data = sig.hilbert(data)
        data = data[:N]
        if self.mode == "Tiempo":
            self.ax.plot(np.real(data), color=self.line_color)
            self.ax.set_xlabel("Muestras", color='white')
            self.ax.set_ylabel("Amplitud", color='white')
        else:
            win = sig.get_window(self.window_type, len(data), fftbins=True)
            d = data * win
            spec = np.fft.fftshift(np.fft.fft(d, n=N))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) + center_hz
            mag = 20 * np.log10(np.abs(spec) + 1e-6)
            span_hz = self.span_mhz * 1e6
            f0 = center_hz - span_hz/2
            f1 = center_hz + span_hz/2
            mask = (freqs >= f0) & (freqs <= f1)
            f_disp = freqs[mask] / 1e6
            mag_disp = mag[mask]
            self.ax.plot(f_disp, mag_disp, color=self.line_color)
            for f in marks:
                self.ax.axvline(f/1e6, color='red', linestyle='--')
            self.ax.set_xlabel("Frecuencia (MHz)", color='white')
            self.ax.set_ylabel("Magnitud (dB)", color='white')
            # --- AUTOSCALE ---
            if not self.auto_scale:
                self.ax.set_ylim(self.ymin_db, self.ymax_db)
            # Si auto_scale está activo, matplotlib ajusta el eje Y automáticamente
        self.ax.set_title(self.title, color='white')
        self.ax.tick_params(colors='white')
        if self.grid_on:
            self.ax.grid(True, color='white', alpha=0.2)
        self.draw()

class WaterfallCanvas(FigureCanvas):
    def __init__(self, fft_size=2048, n_rows=400, colormap="inferno"):
        self.fft_size = fft_size
        self.n_rows = n_rows
        self.colormap = colormap
        self.fs = 48000
        self.center_hz = 0
        self.span_mhz = 2.0
        self.history = np.full((self.n_rows, self.fft_size), -120.0)
        fig = Figure(figsize=(8, 3), facecolor="black")
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.img = self.ax.imshow(
            self.history, aspect="auto", origin="lower",
            extent=[-1, 1, 0, 1], cmap=self.colormap,
            vmin=-120, vmax=0
        )
        self.ax.set_xlabel("Frecuencia (MHz)", color="white")
        self.ax.set_ylabel("Tiempo", color="white")
        self.ax.set_title("Waterfall Dinámico", color="white")
        self.ax.tick_params(colors="white")
        self.ax.grid(True, color="white", alpha=0.2)
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.draw()

    def update_waterfall(self, data, fs, center_hz=0):
        if self.history.shape[1] != self.fft_size:
            self.history = np.full((self.n_rows, self.fft_size), -120.0)
        if len(data) < self.fft_size:
            return
        self.fs, self.center_hz = fs, center_hz
        N = self.fft_size
        x = data[:N]
        if not np.iscomplexobj(x):
            x = sig.hilbert(np.real(x))
        win  = sig.windows.hann(N)
        spec = np.fft.fftshift(np.fft.fft(x * win))
        mag  = 20 * np.log10(np.abs(spec) + 1e-6)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :] = mag
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) + center_hz
        span_hz = self.span_mhz * 1e6
        if span_hz <= 0:
            f0, f1 = freqs.min(), freqs.max()
        else:
            f0 = center_hz - span_hz / 2
            f1 = center_hz + span_hz / 2
        mask = (freqs >= f0) & (freqs <= f1)
        f_disp = freqs[mask] / 1e6
        self.img.set_extent([f_disp[0], f_disp[-1], 0, 1])
        self.img.set_data(self.history[:, mask])
        self.draw()

def scan_peaks(data, fs, center_hz=0):
    N = min(4096, len(data))
    if not np.iscomplexobj(data):
        data = sig.hilbert(data)
    win = sig.get_window('hann', N)
    spec = np.fft.fftshift(np.fft.fft(data[:N] * win))
    mag = 20 * np.log10(np.abs(spec) + 1e-6)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) + center_hz
    peaks, _ = sig.find_peaks(mag, height=np.max(mag)-15, distance=20)
    return freqs[peaks]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Co-Channel Canceller AM/FM for CleanRF")
        self.resize(1400, 900)
        self.fs = 48000
        self.ptr = 0
        self.chunk = 4096
        self.center_hz = 0.0
        self.data = np.array([])
        self._apply_dark_theme()
        self.canceller = CoChannelCanceller(self.fs)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self._build_ui()

    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#121212"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
        self.setPalette(palette)

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        # Top bar
        bar = QHBoxLayout()
        self.lbl_file = QLabel("Archivo no cargado")
        self.btn_load = QPushButton("Cargar WAV")
        self.btn_start = QPushButton("Iniciar")
        self.chk_listen = QCheckBox("Escuchar")
        self.sld_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_vol.setRange(0, 100)
        self.sld_vol.setValue(50)
        for w in [self.lbl_file, self.btn_load, self.btn_start, self.chk_listen, QLabel("Vol"), self.sld_vol]:
            bar.addWidget(w)
        layout.addLayout(bar)
        # Frecuencia central y offset
        freq = QHBoxLayout()
        self.center_box = QDoubleSpinBox()
        self.center_box.setDecimals(6)
        self.center_box.setSuffix(" MHz")
        self.center_box.setRange(0.0, 15000.0)
        self.center_box.setSingleStep(0.1)
        self.center_box.valueChanged.connect(lambda v: setattr(self, "center_hz", v * 1e6))
        freq.addWidget(QLabel("Freq Central"))
        freq.addWidget(self.center_box)
        self.offset_box = QDoubleSpinBox()
        self.offset_box.setDecimals(1)
        self.offset_box.setSuffix(" Hz")
        self.offset_box.setRange(-1e6, 1e6)
        self.offset_box.setSingleStep(1)
        self.offset_box.valueChanged.connect(self.update_offset_bar)
        freq.addWidget(QLabel("Offset"))
        freq.addWidget(self.offset_box)
        layout.addLayout(freq)
        # Cancelador único
        g1 = QGroupBox("Co-Channel Canceller")
        l1 = QVBoxLayout(g1)

        # Fila 1: Modo
        row_mode = QHBoxLayout()
        self.mode_cb = QComboBox(); self.mode_cb.addItems(["AM", "FM"])
        row_mode.addWidget(QLabel("Modo:"))
        row_mode.addWidget(self.mode_cb)
        l1.addLayout(row_mode)

        # Fila 2: Enabled, Remove Carrier, Auto Tune
        row1 = QHBoxLayout()
        self.chk_enable = QCheckBox("Enabled")
        self.chk_remove_carrier = QCheckBox("Remove Carrier")
        self.chk_auto_tune = QCheckBox("Auto Tune")
        row1.addWidget(self.chk_enable)
        row1.addWidget(self.chk_remove_carrier)
        row1.addWidget(self.chk_auto_tune)
        l1.addLayout(row1)

        # Fila 3: Carrier Offset, Bandwidth, IF Offset
        row2 = QHBoxLayout()
        self.carrier_offset = QDoubleSpinBox()
        self.carrier_offset.setDecimals(3)
        self.carrier_offset.setSuffix(" Hz")
        self.carrier_offset.setRange(-1e6, 1e6)
        self.carrier_offset.setSingleStep(1)
        row2.addWidget(QLabel("Carrier Offset"))
        row2.addWidget(self.carrier_offset)
        self.bandwidth = QDoubleSpinBox()
        self.bandwidth.setDecimals(3)
        self.bandwidth.setSuffix(" Hz")
        self.bandwidth.setRange(10, 20000)
        self.bandwidth.setSingleStep(10)
        row2.addWidget(QLabel("Bandwidth"))
        row2.addWidget(self.bandwidth)
        self.if_offset = QDoubleSpinBox()
        self.if_offset.setDecimals(3)
        self.if_offset.setSuffix(" Hz")
        self.if_offset.setRange(-1e6, 1e6)
        self.if_offset.setSingleStep(1)
        row2.addWidget(QLabel("IF Offset"))
        row2.addWidget(self.if_offset)
        l1.addLayout(row2)

        # Fila 4: Correction/Sensitivity
        row3 = QHBoxLayout()
        self.sld_correction = QSlider(Qt.Orientation.Horizontal)
        self.sld_correction.setRange(-20, 20)
        self.sld_correction.setValue(0)
        self.lbl_correction = QLabel("0 dB")
        row3.addWidget(QLabel("Correction"))  # Cambia a "Sensitivity" en FM
        row3.addWidget(self.sld_correction)
        row3.addWidget(self.lbl_correction)
        l1.addLayout(row3)

        layout.addWidget(g1)

        # --- Mostrar/ocultar controles según modo ---
        def update_amfm_controls():
            is_am = self.mode_cb.currentText() == "AM"
            self.chk_remove_carrier.setVisible(is_am)
            self.chk_auto_tune.setVisible(is_am)
            row3.itemAt(0).widget().setText("Correction" if is_am else "Sensitivity")
        self.mode_cb.currentTextChanged.connect(update_amfm_controls)
        update_amfm_controls()

        self.sld_correction.valueChanged.connect(lambda v: self.lbl_correction.setText(f"{v} dB"))

        # Visual config
        vis_cfg = QGroupBox("Visualización")
        vis_l = QHBoxLayout(vis_cfg)
        self.fft_cb = QComboBox(); self.fft_cb.addItems([str(x) for x in [128, 256, 512, 1024, 2048, 4096, 8192]])
        self.win_cb = QComboBox(); self.win_cb.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.co_orig_cb = QComboBox(); self.co_orig_cb.addItems(["cyan", "lime", "yellow", "white"])
        self.co_proc_cb = QComboBox(); self.co_proc_cb.addItems(["orange", "red", "magenta", "white"])
        self.chk_grid = QCheckBox("Grid"); self.chk_grid.setChecked(True)
        self.chk_auto = QCheckBox("Autoscale"); self.chk_auto.setChecked(True)
        self.ymin_s = QDoubleSpinBox(); self.ymin_s.setRange(-200, 200); self.ymin_s.setValue(-100)
        self.ymax_s = QDoubleSpinBox(); self.ymax_s.setRange(-200, 200); self.ymax_s.setValue(0)
        self.span_s = QDoubleSpinBox(); self.span_s.setRange(0.01, 100.0); self.span_s.setValue(2.0)
        for w, lbl in [
            (self.fft_cb, "FFT:"), (self.win_cb, "Ventana:"),
            (self.co_orig_cb, "Color original:"), (self.co_proc_cb, "Color cancelada:"),
            (self.chk_grid, None), (self.chk_auto, None),
            (self.ymin_s, "Ymin:"), (self.ymax_s, "Ymax:"), (self.span_s, "Span (MHz):")
        ]:
            if lbl: vis_l.addWidget(QLabel(lbl))
            vis_l.addWidget(w)
        layout.addWidget(vis_cfg)
        # Visualizadores con selector de modo
        vis_container = QWidget()
        vis_layout = QHBoxLayout(vis_container)
        vis_layout.setContentsMargins(0, 0, 0, 0)
        vis_layout.setSpacing(16)

        # Señal original
        orig_box = QVBoxLayout()
        self.orig_mode_cb = QComboBox()
        self.orig_mode_cb.addItems(["Tiempo", "Frecuencia", "Waterfall"])
        orig_box.addWidget(QLabel("Señal Original"))
        orig_box.addWidget(self.orig_mode_cb)
        self.vis_orig = Visualizer("Señal Original")
        self.wfall1 = WaterfallCanvas()
        orig_box.addWidget(self.vis_orig)
        orig_box.addWidget(self.wfall1)
        vis_layout.addLayout(orig_box)

        # Señal cancelada
        proc_box = QVBoxLayout()
        self.proc_mode_cb = QComboBox()
        self.proc_mode_cb.addItems(["Tiempo", "Frecuencia", "Waterfall"])
        proc_box.addWidget(QLabel("Señal Cancelada"))
        proc_box.addWidget(self.proc_mode_cb)
        self.vis_proc = Visualizer("Señal Cancelada")
        self.wfall2 = WaterfallCanvas()
        proc_box.addWidget(self.vis_proc)
        proc_box.addWidget(self.wfall2)
        vis_layout.addLayout(proc_box)

        layout.addWidget(vis_container)
        # Botón de escaneo
        self.btn_scan = QPushButton("🔍 Escanear portadoras")
        self.btn_scan.clicked.connect(self.scan_and_mark)
        layout.addWidget(self.btn_scan)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)

        # Conexiones para cambiar modo de visualización
        self.orig_mode_cb.currentTextChanged.connect(self.update_view_modes)
        self.proc_mode_cb.currentTextChanged.connect(self.update_view_modes)
        self.update_view_modes()

        # --- AGREGA ESTAS DOS LÍNEAS ---
        self.btn_load.clicked.connect(self.load_wav)
        self.btn_start.clicked.connect(self.start_processing)

    def update_view_modes(self):
        orig_mode = self.orig_mode_cb.currentText()
        proc_mode = self.proc_mode_cb.currentText()
        self.vis_orig.setVisible(orig_mode != "Waterfall")
        self.wfall1.setVisible(orig_mode == "Waterfall")
        self.vis_proc.setVisible(proc_mode != "Waterfall")
        self.wfall2.setVisible(proc_mode == "Waterfall")

    def update_offset_bar(self):
        # Actualiza la barra roja en la gráfica según el offset
        self.update_loop(force=True)

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar archivo WAV", "", "Archivos WAV (*.wav)")
        if not path:
            return
        import soundfile as sf
        try:
            data, fs = sf.read(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")
            return
        if data.ndim == 1:
            data = data.astype(np.float32)
        elif data.ndim == 2:
            if data.shape[1] >= 2:
                data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
            else:
                data = data[:, 0].astype(np.float32)
        else:
            QMessageBox.critical(self, "Error", "Formato de archivo WAV no soportado.")
            return
        if data.size < 1024:
            QMessageBox.critical(self, "Error", "El archivo es demasiado corto.")
            return
        self.data = data
        self.fs = fs
        self.ptr = 0
        self.lbl_file.setText(f"Archivo: {path.split('/')[-1]}")
        self.canceller = CoChannelCanceller(fs)
        self.vis_orig.refresh(np.zeros(2048), self.fs)
        self.vis_proc.refresh(np.zeros(2048), self.fs)
        self.wfall1.update_waterfall(np.zeros(2048), self.fs)
        self.wfall2.update_waterfall(np.zeros(2048), self.fs)

    def start_processing(self):
        if self.data is None or len(self.data) == 0:
            QMessageBox.warning(self, "Error", "No hay archivo cargado.")
            return
        self.ptr = 0
        self.timer.start(40)

    def update_loop(self, force=False):
        if self.data is None or len(self.data) == 0:
            return
        s = self.ptr
        e = s + self.chunk
        if e >= len(self.data):
            s, e = 0, self.chunk
            self.ptr = 0
        self.ptr = e
        x = self.data[s:e]
        # Cancelador
        enabled = self.chk_enable.isChecked()
        remove_carrier = self.chk_remove_carrier.isChecked() if self.mode_cb.currentText() == "AM" else False
        mode = self.mode_cb.currentText()
        offset = self.carrier_offset.value()
        bw = self.bandwidth.value()
        if_offset = self.if_offset.value()
        correction = 10 ** (self.sld_correction.value() / 20)
        total_offset = offset + if_offset
        self.canceller.update(
            total_offset, bw, correction, remove_carrier, enabled, mode
        )
        y = self.canceller.process(x)
        # Audio
        if self.chk_listen.isChecked() and sd is not None:
            if mode == "FM":
                audio = demod_fm(y)
            else:
                audio = demod_am(y)
            audio = (audio / (np.max(np.abs(audio)) + 1e-6)) * (self.sld_vol.value() / 100)
            try:
                sd.play(audio, self.fs, blocking=False)
            except Exception:
                pass
        elif sd:
            try:
                sd.stop()
            except Exception:
                pass
        # Visual config
        fft = int(self.fft_cb.currentText())
        win = self.win_cb.currentText()
        color_orig = self.co_orig_cb.currentText()
        color_proc = self.co_proc_cb.currentText()
        grid = self.chk_grid.isChecked()
        auto = self.chk_auto.isChecked()
        ymin = self.ymin_s.value()
        ymax = self.ymax_s.value()
        span = self.span_s.value()

        # Aplica configuración a los visualizadores
        for vis, color in [(self.vis_orig, color_orig), (self.vis_proc, color_proc)]:
            vis.fft_size = fft
            vis.window_type = win
            vis.line_color = color
            vis.grid_on = grid
            vis.auto_scale = auto
            vis.ymin_db = ymin
            vis.ymax_db = ymax
            vis.span_mhz = span

        # Aplica configuración a los waterfalls
        for wfall in [self.wfall1, self.wfall2]:
            wfall.fft_size = fft
            wfall.span_mhz = span

        # Set modo visualización
        self.vis_orig.mode = self.orig_mode_cb.currentText()
        self.vis_proc.mode = self.proc_mode_cb.currentText()
        # Refresca solo el visualizador visible
        if self.vis_orig.isVisible():
            self.vis_orig.refresh(x, self.fs, self.center_hz, marks=[offset + self.center_hz])
        if self.vis_proc.isVisible():
            self.vis_proc.refresh(y, self.fs, self.center_hz, marks=[offset + self.center_hz])
        if self.wfall1.isVisible():
            self.wfall1.update_waterfall(x, self.fs, self.center_hz)
        if self.wfall2.isVisible():
            self.wfall2.update_waterfall(y, self.fs, self.center_hz)

    def scan_and_mark(self):
        s = self.ptr
        e = min(len(self.data), s + 4096)
        seg = self.data[s:e]
        peaks = scan_peaks(seg, self.fs, center_hz=self.center_hz)
        freqs_str = "\n".join([f"{f/1e6:.3f} MHz" for f in peaks])
        QMessageBox.information(self, "Portadoras detectadas", f"Frecuencias detectadas:\n{freqs_str}")
        self.vis_proc.refresh(seg, self.fs, self.center_hz, marks=peaks)

    def update_gain_label(self, value):
        self.lbl_gain.setText(f"{value} dB")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
