import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import struct
import wave
import math

# Verificar e importar dependencias paso a paso
import_errors = []
NUMPY_AVAILABLE = False
SCIPY_AVAILABLE = False  
MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✓ NumPy imported successfully")
except ImportError as e:
    import_errors.append(f"NumPy: {e}")

try:
    import scipy.signal as signal
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
    print("✓ SciPy imported successfully")
except ImportError as e:
    import_errors.append(f"SciPy: {e}")

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Forzar backend TkAgg
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvasTkinter
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    import_errors.append(f"Matplotlib: {e}")
    try:
        # Fallback: intentar con backend diferente
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        print("⚠ Matplotlib imported with Agg backend (no GUI)")
    except ImportError:
        pass

FULL_FEATURES = NUMPY_AVAILABLE and SCIPY_AVAILABLE and MATPLOTLIB_AVAILABLE

if import_errors:
    print("Import issues found:")
    for error in import_errors:
        print(f"  - {error}")
        
if not FULL_FEATURES:
    print(f"\nFeature status:")
    print(f"  NumPy: {'✓' if NUMPY_AVAILABLE else '✗'}")
    print(f"  SciPy: {'✓' if SCIPY_AVAILABLE else '✗'}")  
    print(f"  Matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'}")
    print(f"  Full Features: {'✓' if FULL_FEATURES else '✗'}")
    
    # Implementaciones básicas sin numpy
    class np:
        @staticmethod
        def array(data):
            return list(data)
        
        @staticmethod
        def zeros(size):
            return [0.0] * size
        
        @staticmethod
        def zeros_like(data):
            return [0.0] * len(data)
        
        @staticmethod
        def sin(x):
            if isinstance(x, list):
                return [math.sin(i) for i in x]
            return math.sin(x)
        
        @staticmethod
        def cos(x):
            if isinstance(x, list):
                return [math.cos(i) for i in x]
            return math.cos(x)
        
        @staticmethod
        def pi():
            return math.pi
        
        @staticmethod
        def arange(n):
            return list(range(int(n)))
        
        @staticmethod
        def max(data):
            return max(data)
        
        @staticmethod
        def abs(data):
            if isinstance(data, list):
                return [abs(x) for x in data]
            return abs(data)
        
        @staticmethod
        def mean(data, axis=None):
            if axis == 1 and isinstance(data[0], list):
                return [(sum(row) / len(row)) for row in data]
            return sum(data) / len(data)
        
        float32 = float
        int16 = int

class SDRCoChannelCanceller:
    def __init__(self, root):
        self.root = root
        self.root.title("SDR Co-Channel Canceller")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.original_data = None
        self.processed_data = None
        self.sample_rate = None
        self.modulation_type = tk.StringVar(value="AM")
        self.interference_freq = tk.DoubleVar(value=1000.0)
        self.adaptation_rate = tk.DoubleVar(value=0.01)
        self.filter_order = tk.IntVar(value=64)
        self.cancellation_depth = tk.DoubleVar(value=20.0)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Estilo moderno
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#404040', foreground='white')
        style.configure('TFrame', background='#2b2b2b')
        
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel de control izquierdo
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Mostrar estado de dependencias
        if not FULL_FEATURES:
            warning_frame = ttk.LabelFrame(control_frame, text="⚠️ Warning", padding=5)
            warning_frame.pack(fill=tk.X, pady=(0, 10))
            ttk.Label(warning_frame, text="Limited features available.\nInstall: pip install numpy scipy matplotlib", 
                     foreground='orange', wraplength=200).pack()
        
        # Cargar archivo
        load_frame = ttk.Frame(control_frame)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="📁 Load WAV File", 
                  command=self.load_wav_file).pack(fill=tk.X)
        
        # Configuración de modulación
        mod_frame = ttk.LabelFrame(control_frame, text="Modulation Type", padding=5)
        mod_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(mod_frame, text="AM", variable=self.modulation_type, 
                       value="AM").pack(anchor=tk.W)
        ttk.Radiobutton(mod_frame, text="FM", variable=self.modulation_type, 
                       value="FM").pack(anchor=tk.W)
        
        # Parámetros de cancelación
        params_frame = ttk.LabelFrame(control_frame, text="Cancellation Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frecuencia de interferencia
        ttk.Label(params_frame, text="Interference Freq (Hz):").pack(anchor=tk.W)
        interference_scale = ttk.Scale(params_frame, from_=100, to=10000, 
                                     variable=self.interference_freq, orient=tk.HORIZONTAL)
        interference_scale.pack(fill=tk.X)
        
        freq_label = ttk.Label(params_frame, text="")
        freq_label.pack(anchor=tk.W)
        
        def update_freq_label(*args):
            freq_label.config(text=f"{self.interference_freq.get():.1f} Hz")
        self.interference_freq.trace('w', update_freq_label)
        update_freq_label()
        
        # Tasa de adaptación
        ttk.Label(params_frame, text="Adaptation Rate:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(params_frame, from_=0.001, to=0.1, 
                 variable=self.adaptation_rate, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Orden del filtro
        ttk.Label(params_frame, text="Filter Order:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(params_frame, from_=16, to=128, 
                 variable=self.filter_order, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Profundidad de cancelación
        ttk.Label(params_frame, text="Cancellation Depth (dB):").pack(anchor=tk.W, pady=(10, 0))
        ttk.Scale(params_frame, from_=10, to=50, 
                 variable=self.cancellation_depth, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Botones de procesamiento
        process_frame = ttk.Frame(control_frame)
        process_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(process_frame, text="🔄 Process Signal", 
                  command=self.process_signal).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(process_frame, text="💾 Save Result", 
                  command=self.save_result).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(process_frame, text="▶️ Play Original", 
                  command=self.play_original).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(process_frame, text="▶️ Play Processed", 
                  command=self.play_processed).pack(fill=tk.X)
        
        # Panel de visualización derecho (solo si matplotlib está disponible)
        if MATPLOTLIB_AVAILABLE:
            viz_frame = ttk.Frame(main_frame)
            viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Crear figura de matplotlib
            self.fig = Figure(figsize=(10, 8), facecolor='#2b2b2b')
            self.canvas = FigureCanvasTkinter(self.fig, viz_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            # Panel informativo cuando no hay matplotlib
            info_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
            info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            info_text = "📊 Visualization Status:\n\n"
            info_text += f"NumPy: {'✓' if NUMPY_AVAILABLE else '✗ pip install numpy'}\n"
            info_text += f"SciPy: {'✓' if SCIPY_AVAILABLE else '✗ pip install scipy'}\n"
            info_text += f"Matplotlib: {'✗ Backend issue' if not MATPLOTLIB_AVAILABLE else '✓'}\n\n"
            
            if not MATPLOTLIB_AVAILABLE:
                info_text += "Try:\n"
                info_text += "pip install matplotlib\n"
                info_text += "or\n"
                info_text += "conda install matplotlib\n\n"
                info_text += "Audio processing still works!"
            
            ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(expand=True)
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Ready - Load a WAV file to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_wav_file(self):
        """Cargar archivo WAV"""
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if SCIPY_AVAILABLE:
                    self.sample_rate, self.original_data = wavfile.read(file_path)
                    
                    # Convertir a mono si es estéreo
                    if len(self.original_data.shape) > 1:
                        self.original_data = np.mean(self.original_data, axis=1)
                    
                    # Normalizar
                    self.original_data = self.original_data.astype(np.float32)
                    self.original_data = self.original_data / np.max(np.abs(self.original_data))
                else:
                    # Implementación básica sin scipy
                    with wave.open(file_path, 'rb') as wav_file:
                        self.sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(-1)
                        
                        # Convertir bytes a números
                        if wav_file.getsampwidth() == 2:  # 16-bit
                            data = struct.unpack(f'<{len(frames)//2}h', frames)
                        else:  # 8-bit
                            data = struct.unpack(f'{len(frames)}b', frames)
                        
                        # Convertir a mono si es estéreo
                        if wav_file.getnchannels() == 2:
                            data = [data[i] + data[i+1] for i in range(0, len(data), 2)]
                        
                        # Normalizar
                        max_val = max(abs(x) for x in data)
                        self.original_data = [float(x) / max_val for x in data]
                
                self.status_var.set(f"Loaded: {os.path.basename(file_path)} "
                                  f"({self.sample_rate} Hz, {len(self.original_data)} samples)")
                
                if MATPLOTLIB_AVAILABLE:
                    self.plot_original_signal()
                else:
                    messagebox.showinfo("Info", "File loaded successfully!\nVisualization requires matplotlib with TkAgg backend.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}\n\nTry installing: pip install numpy scipy matplotlib")
    
    def am_cochannel_canceller(self, signal_data, interference_freq, sample_rate, 
                              adaptation_rate, filter_order, depth_db):
        """Algoritmo de cancelación co-canal para AM"""
        # Generar señal de referencia basada en la frecuencia de interferencia
        t = np.arange(len(signal_data)) / sample_rate
        reference = np.sin(2 * np.pi * interference_freq * t)
        
        # Filtro adaptativo LMS (Least Mean Squares)
        w = np.zeros(filter_order)  # Coeficientes del filtro
        output = np.zeros_like(signal_data)
        
        # Buffer para la señal de referencia
        ref_buffer = np.zeros(filter_order)
        
        for n in range(len(signal_data)):
            # Actualizar buffer de referencia
            ref_buffer[1:] = ref_buffer[:-1]
            ref_buffer[0] = reference[n]
            
            # Calcular salida del filtro adaptativo
            y = np.dot(w, ref_buffer)
            
            # Error (señal deseada - interferencia estimada)
            error = signal_data[n] - y
            output[n] = error
            
            # Actualizar coeficientes del filtro (algoritmo LMS)
            w += adaptation_rate * error * ref_buffer
        
        # Aplicar control de profundidad
        attenuation = 10 ** (-depth_db / 20)
        cancelled_interference = signal_data - output
        output = signal_data - attenuation * cancelled_interference
        
        return output
    
    def fm_cochannel_canceller(self, signal_data, interference_freq, sample_rate, 
                              adaptation_rate, filter_order, depth_db):
        """Algoritmo de cancelación co-canal para FM"""
        # Para FM, trabajamos con la señal analítica (compleja)
        analytic_signal = signal.hilbert(signal_data)
        
        # Generar señal de referencia compleja
        t = np.arange(len(signal_data)) / sample_rate
        reference_i = np.sin(2 * np.pi * interference_freq * t)
        reference_q = np.cos(2 * np.pi * interference_freq * t)
        
        # Filtros adaptativos para componentes I y Q
        w_i = np.zeros(filter_order)
        w_q = np.zeros(filter_order)
        output = np.zeros_like(analytic_signal, dtype=complex)
        
        # Buffers para las señales de referencia
        ref_i_buffer = np.zeros(filter_order)
        ref_q_buffer = np.zeros(filter_order)
        
        for n in range(len(signal_data)):
            # Actualizar buffers de referencia
            ref_i_buffer[1:] = ref_i_buffer[:-1]
            ref_i_buffer[0] = reference_i[n]
            ref_q_buffer[1:] = ref_q_buffer[:-1]
            ref_q_buffer[0] = reference_q[n]
            
            # Calcular salidas de los filtros adaptativos
            y_i = np.dot(w_i, ref_i_buffer)
            y_q = np.dot(w_q, ref_q_buffer)
            
            # Error complejo
            error = analytic_signal[n] - (y_i + 1j * y_q)
            output[n] = error
            
            # Actualizar coeficientes de los filtros
            w_i += adaptation_rate * error.real * ref_i_buffer
            w_q += adaptation_rate * error.imag * ref_q_buffer
        
        # Extraer parte real y aplicar control de profundidad
        real_output = output.real
        attenuation = 10 ** (-depth_db / 20)
        cancelled_interference = signal_data - real_output
        final_output = signal_data - attenuation * cancelled_interference
        
        return final_output
    
    def process_signal(self):
        """Procesar la señal con cancelación co-canal"""
        if self.original_data is None:
            messagebox.showwarning("Warning", "Please load a WAV file first")
            return
        
        self.status_var.set("Processing signal...")
        
        def process_thread():
            try:
                if self.modulation_type.get() == "AM":
                    self.processed_data = self.am_cochannel_canceller(
                        self.original_data,
                        self.interference_freq.get(),
                        self.sample_rate,
                        self.adaptation_rate.get(),
                        self.filter_order.get(),
                        self.cancellation_depth.get()
                    )
                else:  # FM
                    self.processed_data = self.fm_cochannel_canceller(
                        self.original_data,
                        self.interference_freq.get(),
                        self.sample_rate,
                        self.adaptation_rate.get(),
                        self.filter_order.get(),
                        self.cancellation_depth.get()
                    )
                
                # Normalizar salida
                self.processed_data = self.processed_data / np.max(np.abs(self.processed_data))
                
                # Actualizar visualización en el hilo principal
                self.root.after(0, self.plot_comparison)
                self.root.after(0, lambda: self.status_var.set("Processing completed"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
                self.root.after(0, lambda: self.status_var.set("Processing failed"))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def plot_original_signal(self):
        """Visualizar señal original"""
        self.fig.clear()
        
        # Crear subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        
        # Configurar colores para tema oscuro
        self.fig.patch.set_facecolor('#2b2b2b')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Tiempo
        t = np.arange(len(self.original_data)) / self.sample_rate
        
        # Señal en tiempo
        ax1.plot(t[:min(10000, len(t))], 
                self.original_data[:min(10000, len(self.original_data))], 
                color='cyan', linewidth=0.8)
        ax1.set_title('Original Signal - Time Domain')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Espectro de frecuencias
        f, Pxx = signal.welch(self.original_data, self.sample_rate, nperseg=1024)
        ax2.semilogy(f, Pxx, color='orange', linewidth=1)
        ax2.set_title('Original Signal - Frequency Spectrum')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def plot_comparison(self):
        """Visualizar comparación de señales"""
        if self.processed_data is None:
            return
        
        self.fig.clear()
        
        # Crear subplots
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)
        
        # Configurar colores para tema oscuro
        self.fig.patch.set_facecolor('#2b2b2b')
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Tiempo
        t = np.arange(len(self.original_data)) / self.sample_rate
        plot_samples = min(5000, len(t))
        
        # Señal original - tiempo
        ax1.plot(t[:plot_samples], self.original_data[:plot_samples], 
                color='cyan', linewidth=0.8)
        ax1.set_title('Original Signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Señal procesada - tiempo
        ax2.plot(t[:plot_samples], self.processed_data[:plot_samples], 
                color='lime', linewidth=0.8)
        ax2.set_title('Processed Signal (Co-Channel Cancelled)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Espectro original
        f, Pxx_orig = signal.welch(self.original_data, self.sample_rate, nperseg=1024)
        ax3.semilogy(f, Pxx_orig, color='orange', linewidth=1, label='Original')
        ax3.set_title('Frequency Spectrum Comparison')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('PSD')
        ax3.grid(True, alpha=0.3)
        
        # Espectro procesado
        f, Pxx_proc = signal.welch(self.processed_data, self.sample_rate, nperseg=1024)
        ax3.semilogy(f, Pxx_proc, color='lime', linewidth=1, label='Processed')
        ax3.legend()
        
        # Diferencia (interferencia cancelada)
        difference = self.original_data - self.processed_data
        ax4.plot(t[:plot_samples], difference[:plot_samples], 
                color='red', linewidth=0.8)
        ax4.set_title('Cancelled Interference')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_result(self):
        """Guardar resultado procesado"""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "No processed data to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save processed audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convertir a int16 para guardar
                output_data = (self.processed_data * 32767).astype(np.int16)
                wavfile.write(file_path, self.sample_rate, output_data)
                self.status_var.set(f"Saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def play_original(self):
        """Reproducir audio original (placeholder)"""
        messagebox.showinfo("Info", "Audio playback would be implemented here.\n"
                          "Consider using pygame or playsound library for actual playback.")
    
    def play_processed(self):
        """Reproducir audio procesado (placeholder)"""
        if self.processed_data is None:
            messagebox.showwarning("Warning", "No processed data to play")
            return
        messagebox.showinfo("Info", "Processed audio playback would be implemented here.\n"
                          "Consider using pygame or playsound library for actual playback.")

def main():
    root = tk.Tk()
    app = SDRCoChannelCanceller(root)
    root.mainloop()

if __name__ == "__main__":
    main()