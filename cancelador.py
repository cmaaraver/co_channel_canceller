#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, numpy as np, soundfile as sf, scipy.signal as sig
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QCheckBox,
    QComboBox, QDoubleSpinBox, QMessageBox, QGroupBox, QScrollArea,
    QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# sounddevice opcional
try:
    import sounddevice as sd
except ImportError:
    sd = None

# -----------------------------------------------------------------------------
def make_notch(w0, r=0.98):
    b = [1, -2*np.cos(w0), 1]
    a = [1, -2*r*np.cos(w0), r*r]
    return b, a

def demod_am(x):
    env = np.abs(sig.hilbert(np.real(x)))
    return env - np.mean(env)

def demod_fm(x):
    ph = np.angle(x)
    dph = np.diff(np.unwrap(ph), prepend=ph[0])
    return dph

class FMCoChannelCanceller:
    def __init__(self, fs):
        self.fs, self.cancel_on = fs, True

    def update(self, offset, bw, gain, do_notch, enabled):
        self.offset, self.bw, self.gain = offset, bw, gain
        self.do_notch, self.cancel_on = do_notch, enabled
        self.mu = 0.01 * gain**2
        self.N = max(16, min(512, int(self.fs/self.bw)))
        self.w = np.zeros(self.N, dtype=complex)
        self.buf = np.zeros(self.N, dtype=complex)
        w0 = 2*np.pi*offset/self.fs
        self.b_notch, self.a_notch = make_notch(w0)

    def process(self, data):
        if not self.cancel_on or len(data)==0:
            return data.real if np.iscomplexobj(data) else data.copy()
        t = np.arange(len(data)) / self.fs
        ref = np.exp(1j*2*np.pi*self.offset*t)
        x = data * ref
        if self.do_notch:
            x = sig.lfilter(self.b_notch, self.a_notch, x)
        y = np.zeros_like(data, dtype=float)
        for n in range(len(data)):
            self.buf = np.roll(self.buf, -1)
            self.buf[-1] = x[n]
            est = np.vdot(self.w, self.buf)
            err = data[n] - self.gain*est
            self.w += self.mu * err * self.buf
            y[n] = err.real
        return y

class AMCoChannelCanceller:
    def __init__(self, fs):
        self.fs, self.cancel_on = fs, True

    def update(self, offset, bw, gain, do_notch, enabled):
        self.offset, self.bw, self.gain = offset, bw, gain
        self.do_notch, self.cancel_on = do_notch, enabled
        self.mu = 0.01 * gain**2
        self.N = max(16, min(512, int(self.fs/self.bw)))
        self.w = np.zeros(self.N)
        self.buf = np.zeros(self.N)
        w0 = 2*np.pi*offset/self.fs
        self.b_notch, self.a_notch = make_notch(w0)

    def process(self, data):
        if not self.cancel_on or len(data)==0:
            return data.copy()
        t = np.arange(len(data)) / self.fs
        ref = np.cos(2*np.pi*self.offset*t)
        x = data * ref
        if self.do_notch:
            x = sig.lfilter(self.b_notch, self.a_notch, x)
        y = np.zeros_like(data)
        for n in range(len(data)):
            self.buf = np.roll(self.buf, -1)
            self.buf[-1] = x[n]
            est = np.dot(self.w, self.buf)
            err = data[n] - self.gain*est
            self.w += self.mu * err * self.buf
            y[n] = err
        return y

class Visualizer(FigureCanvas):
    def __init__(self, title=""):
        fig = Figure(facecolor="black"); super().__init__(fig)
        self.ax = fig.add_subplot(111, facecolor="black")
        self.title = title; self.mode = 'time'; self.fft_size = 1024
        self.window_type = 'hann'  # Cambiado aquí
        self.grid_on = True; self.axis_on = True
        self.autoscale = True; self.ymin, self.ymax = -100, 0
        self.ref_level = 0.0; self.hist_len = 100; self.history = None
        self.line_color = 'cyan'; self._theme()

    def _theme(self):
        ax = self.ax
        ax.set_facecolor("black"); ax.title.set_color('white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('white')

    def set_mode(self, m): self.mode = m
    def set_line_color(self, c): self.line_color = c
    def set_settings(self, fft, win, grid, axis, auto, ymin, ymax, ref):
        self.fft_size, self.window_type = fft, win  # Cambiado aquí
        self.grid_on, self.axis_on, self.autoscale = grid, axis, auto
        self.ymin, self.ymax, self.ref_level = ymin, ymax, ref

    def refresh(self, data, fs, center_hz=0, mark_hz=None):
        if data is None or len(data)==0: return
        self.ax.clear(); self._theme()
        N = self.fft_size
        d = np.zeros(N, dtype=complex)
        d[:min(len(data),N)] = data[:min(len(data),N)]
        if not np.iscomplexobj(d): d = sig.hilbert(d)
        d *= sig.get_window(self.window_type, N, fftbins=True)  # Cambiado aquí

        if self.mode=='time':
            t = np.arange(len(data))/fs
            self.ax.plot(t*1e3, np.real(data), color=self.line_color)
            if self.axis_on:
                self.ax.set_xlabel("Tiempo (ms)"); self.ax.set_ylabel("Amplitud")
        else:
            spec = np.fft.fftshift(np.fft.fft(d))
            freqs = np.fft.fftshift(np.fft.fftfreq(N,1/fs))
            mag = 20*np.log10(np.abs(spec)+1e-6)
            xax = freqs/1e6 + center_hz/1e6
            if self.mode=='freq':
                self.ax.plot(xax, mag, color=self.line_color)
            else:
                if self.history is None or self.history.shape[1]!=N:
                    self.history = np.zeros((self.hist_len,N))
                self.history = np.roll(self.history,-1,axis=0)
                self.history[-1,:] = mag
                self.ax.imshow(self.history, aspect='auto', cmap='plasma',
                               origin='upper', extent=[xax[0],xax[-1],0,self.hist_len])
            if mark_hz is not None:
                self.ax.axvline(mark_hz/1e6, color='red', linestyle='--')
            if self.axis_on:
                self.ax.set_xlabel("Frecuencia (MHz)")
                if self.mode=='freq': self.ax.set_ylabel("Magnitud (dB)")
            self.ax.axhline(self.ref_level, color='white', linestyle=':')
            if self.autoscale:
                self.ax.autoscale_view()
            else:
                self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.grid(self.grid_on, color='white', alpha=0.2)
        self.ax.set_title(self.title)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clean RF Canceller + Audio")
        self.resize(1200,960)
        self.data = np.array([]); self.fs=48000; self.ptr=0; self.chunk=2048
        self.fm = FMCoChannelCanceller(self.fs); self.am = AMCoChannelCanceller(self.fs)
        self.timer = QTimer(self); self.timer.timeout.connect(self.update_loop)
        self._apply_theme(); self._build_ui(); self.update_view_settings()

    def _apply_theme(self):
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#121212"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
        self.setPalette(pal)
        self.setStyleSheet("""
            QLabel,QCheckBox,QComboBox,QPushButton,QDoubleSpinBox,QSlider {
              color:white; background:#2c2c2c;
            }
            QSlider::groove:horizontal{background:#444;height:6px;}
            QSlider::handle:horizontal{background:white;width:10px;margin:-5px 0;}
            QGroupBox{color:white;border:1px solid #444;margin-top:10px;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 3px;}
            QCheckBox::indicator{width:50px;height:25px;border-radius:12px;background:#666;}
            QCheckBox::indicator:checked{background:#44c767;}
        """)

    def _build_ui(self):
        c=QWidget(); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(c)
        self.setCentralWidget(scroll); v=QVBoxLayout(c)

        # Top bar + Config
        h1=QHBoxLayout()
        self.load_btn=QPushButton("📂 Cargar WAV")
        self.start_btn=QPushButton("▶ Iniciar")
        self.stop_btn=QPushButton("⏹ Detener")
        self.mode_cb=QComboBox(); self.mode_cb.addItems(["FM","AM"])
        self.mode_cb.currentTextChanged.connect(self._switch_config_page)
        self.config_btn=QPushButton("⚙️ Config"); self.config_btn.setCheckable(True)
        self.config_btn.toggled.connect(self._toggle_config_panel)
        self.view_cb=QComboBox(); self.view_cb.addItems(["time","freq","waterfall"])
        self.cancel_chk=QCheckBox("Cancel ON"); self.cancel_chk.setChecked(True)
        self.listen_chk=QCheckBox("🎧 Listen"); self.listen_chk.setChecked(False)
        self.vol_sld = QSlider(Qt.Orientation.Horizontal); self.vol_sld.setRange(0,100); self.vol_sld.setValue(50)
        self.file_lbl=QLabel("Archivo: (ninguno)")
        for w in [self.load_btn,self.start_btn,self.stop_btn,
                  QLabel("Modo:"),self.mode_cb,self.config_btn,
                  QLabel("Vista:"),self.view_cb,
                  self.cancel_chk,self.listen_chk,
                  QLabel("Vol:"),self.vol_sld,self.file_lbl]:
            h1.addWidget(w)
        h1.addStretch(); v.addLayout(h1)

        # Config panel
        self.config_panel=QWidget(); cfg_v=QVBoxLayout(self.config_panel)
        self.stacker=QStackedWidget()
        self._build_fm_page(); self._build_am_page()
        cfg_v.addWidget(self.stacker)
        self.config_panel.setVisible(False)
        v.addWidget(self.config_panel)

        # Visual config
        gb=QGroupBox("Configuración Visual"); h3=QHBoxLayout(gb)
        self.chk_grid=QCheckBox("Grid"); self.chk_axis=QCheckBox("Ejes")
        self.chk_auto=QCheckBox("Autoscale"); self.chk_grid.setChecked(True)
        self.chk_axis.setChecked(True); self.chk_auto.setChecked(True)
        self.ymin_s=QDoubleSpinBox();self.ymin_s.setRange(-200,200);self.ymin_s.setValue(-100)
        self.ymax_s=QDoubleSpinBox();self.ymax_s.setRange(-200,200);self.ymax_s.setValue(0)
        self.ref_s=QDoubleSpinBox();self.ref_s.setRange(-200,200);self.ref_s.setValue(0)
        self.fft_cb=QComboBox();self.fft_cb.addItems(["512","1024","2048","4096"])
        self.win_cb=QComboBox();self.win_cb.addItems(["rectangular","hann","hamming","blackman"])
        self.co_o_cb=QComboBox();self.co_o_cb.addItems(["cyan","lime","red","magenta","yellow","white"])
        self.co_c_cb=QComboBox();self.co_c_cb.addItems(["orange","white","red","blue","green","magenta"])
        self.co_o_cb.setCurrentText("cyan"); self.co_c_cb.setCurrentText("orange")
        for w,lbl in [(self.chk_grid,None),(self.chk_axis,None),(self.chk_auto,None),
                      (self.ymin_s,"Ymin:"),(self.ymax_s,"Ymax:"),(self.ref_s,"Ref:"),
                      (self.fft_cb,"FFT:"),(self.win_cb,"Ventana:"),
                      (self.co_o_cb,"Color orig."),(self.co_c_cb,"Color can.")]:
            if lbl: h3.addWidget(QLabel(lbl))
            h3.addWidget(w)
        v.addWidget(gb)

        # Visualizers
        self.vis_o=Visualizer("🔵 Original"); self.vis_c=Visualizer("🟠 Cancelada")
        v.addWidget(self.vis_o); v.addWidget(self.vis_c)

        # Signals
        self.load_btn.clicked.connect(self.load_wav)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.timer.stop)
        self.view_cb.currentTextChanged.connect(lambda m: (
            self.vis_o.set_mode(m), self.vis_c.set_mode(m)
        ))
        for w in [self.chk_grid,self.chk_axis,self.chk_auto,
                  self.ymin_s,self.ymax_s,self.ref_s,
                  self.fft_cb,self.win_cb,self.co_o_cb,self.co_c_cb]:
            if isinstance(w, QCheckBox): w.stateChanged.connect(self.update_view_settings)
            elif isinstance(w, QComboBox): w.currentTextChanged.connect(self.update_view_settings)
            else: w.valueChanged.connect(self.update_view_settings)

    def _build_fm_page(self):
        fm_w=QWidget(); fml=QHBoxLayout(fm_w)
        self.fm_center=QDoubleSpinBox(); self.fm_center.setRange(0,6000)
        self.fm_center.setDecimals(3); self.fm_center.setValue(100)
        self.fm_off=QDoubleSpinBox(); self.fm_off.setRange(0,6000)
        self.fm_off.setDecimals(3); self.fm_off.setValue(0)
        self.fm_bw=QDoubleSpinBox(); self.fm_bw.setRange(100,200000)
        self.fm_bw.setSingleStep(100); self.fm_bw.setValue(12000)
        self.fm_gain = QSlider(Qt.Orientation.Horizontal); self.fm_gain.setRange(-60,20); self.fm_gain.setValue(0)
        self.fm_gain_lbl=QLabel("0 dB")
        self.fm_gain.valueChanged.connect(lambda v: self.fm_gain_lbl.setText(f"{v} dB"))
        self.fm_notch=QCheckBox("Remove Carrier"); self.fm_notch.setChecked(False)
        for w,lbl in [(self.fm_center,"Center MHz:"),(self.fm_off,"Offset MHz:"),
                      (self.fm_bw,"BW Hz:"),(self.fm_gain,"Gain dB:"),(self.fm_gain_lbl,None),
                      (self.fm_notch,None)]:
            if lbl: fml.addWidget(QLabel(lbl))
            fml.addWidget(w)
        self.stacker.addWidget(fm_w)

    def _build_am_page(self):
        am_w=QWidget(); aml=QHBoxLayout(am_w)
        self.am_notch=QCheckBox("Remove Carrier"); self.am_notch.setChecked(True)
        self.am_autot=QCheckBox("Auto Tune"); self.am_autot.setChecked(False)
        self.am_off=QDoubleSpinBox(); self.am_off.setRange(0,6000)
        self.am_off.setDecimals(3); self.am_off.setValue(0)
        self.am_bw=QDoubleSpinBox(); self.am_bw.setRange(100,200000)
        self.am_bw.setSingleStep(100); self.am_bw.setValue(5130)
        self.am_if=QDoubleSpinBox(); self.am_if.setRange(0,6000)
        self.am_if.setDecimals(3); self.am_if.setValue(0)
        self.am_corr = QSlider(Qt.Orientation.Horizontal); self.am_corr.setRange(0,100); self.am_corr.setValue(50)
        self.am_corr_lbl=QLabel("50%")
        self.am_corr.valueChanged.connect(lambda v: self.am_corr_lbl.setText(f"{v}%"))
        for w,lbl in [(self.am_notch,None),(self.am_autot,None),
                      (self.am_off,"Carrier MHz:"),(self.am_bw,"BW Hz:"),
                      (self.am_if,"IF MHz:"),(self.am_corr,"Corr %:"),(self.am_corr_lbl,None)]:
            if lbl: aml.addWidget(QLabel(lbl))
            aml.addWidget(w)
        self.stacker.addWidget(am_w)

    def _toggle_config_panel(self, checked):
        self.config_panel.setVisible(checked)
        self._switch_config_page(self.mode_cb.currentText())

    def _switch_config_page(self, mode):
        idx = 0 if mode=="FM" else 1
        self.stacker.setCurrentIndex(idx)
        self.config_panel.setVisible(self.config_btn.isChecked())

    def update_view_settings(self,*_):
        fft=int(self.fft_cb.currentText()); win=self.win_cb.currentText()
        grid=self.chk_grid.isChecked(); axis=self.chk_axis.isChecked()
        auto=self.chk_auto.isChecked()
        ymin,ymax,ref=self.ymin_s.value(),self.ymax_s.value(),self.ref_s.value()
        co,cc=self.co_o_cb.currentText(),self.co_c_cb.currentText()
        for vis,col in [(self.vis_o,co),(self.vis_c,cc)]:
            vis.set_settings(fft,win,grid,axis,auto,ymin,ymax,ref)
            vis.set_line_color(col)

    def load_wav(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Seleccionar WAV","","Archivos WAV (*.wav *.WAV);;Todos (*)"
        )
        if not path: return
        try:
            data,fs=sf.read(path)
        except Exception as e:
            QMessageBox.critical(self,"Error lectura WAV",str(e)); return
        if data.ndim>1 and data.shape[1]>=2:
            data=data[:,0]+1j*data[:,1]
        elif data.ndim>1:
            data=data[:,0]
        if data.size==0:
            QMessageBox.warning(self,"Error","WAV vacío"); return
        self.data,self.fs,self.ptr=data,fs,0
        self.fm=FMCoChannelCanceller(self.fs); self.am=AMCoChannelCanceller(self.fs)
        self.file_lbl.setText("Archivo: "+path.split("/")[-1])

    def on_start(self):
        if len(self.data)==0:
            QMessageBox.warning(self,"Error","Carga primero un WAV."); return
        self.ptr=0; self.timer.start(40)

    def update_loop(self):
        s,e=self.ptr,self.ptr+self.chunk
        if e>len(self.data): s,e,self.ptr=0,self.chunk,0
        seg=self.data[s:e]; self.ptr+=self.chunk
        if self.mode_cb.currentText()=="FM":
            cen=self.fm_center.value()*1e6
            off=self.fm_off.value()*1e6; bw=self.fm_bw.value()
            gain=10**(self.fm_gain.value()/20)
            notch=self.fm_notch.isChecked()
            en=self.cancel_chk.isChecked()
            self.fm.update(off,bw,gain,notch,en)
            proc=self.fm.process(seg); mark=cen+off
        else:
            cen=self.am_if.value()*1e6
            off=self.am_off.value()*1e6; bw=self.am_bw.value()
            gain=self.am_corr.value()/100
            notch=self.am_notch.isChecked()
            en=self.cancel_chk.isChecked()
            self.am.update(off,bw,gain,notch,en)
            proc=self.am.process(seg); mark=cen+off

        self.vis_o.refresh(seg,self.fs,cen)
        self.vis_c.refresh(proc,self.fs,cen,mark)

        if sd and self.listen_chk.isChecked():
            audio = demod_fm(seg) if self.mode_cb.currentText()=="FM" else demod_am(seg)
            mx=np.max(np.abs(audio))+1e-6
            audio=(audio/mx)*(self.vol_sld.value()/100)
            try: sd.play(audio,self.fs,blocking=False)
            except: pass
        elif sd:
            try: sd.stop()
            except: pass

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=MainWindow(); win.show()
    sys.exit(app.exec())
