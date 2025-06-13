#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys
import signal
import faulthandler
faulthandler.enable()

import numpy as np
import uhd
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

from PyQt6.QtCore import (
    QSize, Qt, QThread, QObject, QTimer,
    pyqtSignal, pyqtSlot
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QSlider, QLabel, QHBoxLayout, QVBoxLayout,
    QPushButton, QComboBox
)
import pyqtgraph as pg

# Global defaults
fft_size          = 4096
num_rows          = 200
center_freq       = 750e6
sample_rates      = []
sample_rate       = None
time_plot_samples = 500
gain              = 50

class SDRWorker(QObject):
    time_plot_update      = pyqtSignal(np.ndarray)
    freq_plot_update      = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    error_signal          = pyqtSignal(str)
    # Add signal for thread-safe cleanup
    cleanup_finished      = pyqtSignal()

    def __init__(self, serial, type_):
        super().__init__()
        self.serial      = serial
        self.type        = type_.lower()
        self.sdr         = None
        self.streamer    = None
        self.metadata    = None
        self.recv_buffer = np.zeros((1, fft_size), dtype=np.complex64)
        self.spectrogram = -50 * np.ones((fft_size, num_rows))
        self.PSD_avg     = -50 * np.ones(fft_size)
        self.running     = False
        
        # Timer will be created in the worker thread
        self.acq_timer = None

    @pyqtSlot()
    def setup_sdr(self):
        # Create timer in the worker thread
        self.acq_timer = QTimer()
        self.acq_timer.timeout.connect(self.acquire_data)
        self.acq_timer.setInterval(50)  # 20 FPS
        
        try:
            if self.type == 'usrp':
                self.sdr = uhd.usrp.MultiUSRP(args=f"serial={self.serial}")
                self.sdr.set_rx_rate(sample_rate, 0)
                self.sdr.set_rx_freq(
                    uhd.libpyuhd.types.tune_request(center_freq), 0
                )
                self.sdr.set_rx_gain(gain, 0)

                st_args = uhd.usrp.StreamArgs("fc32", "sc16")
                st_args.channels = [0]
                self.metadata = uhd.types.RXMetadata()
                self.streamer = self.sdr.get_rx_stream(st_args)

                cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                cmd.stream_now = True
                self.streamer.issue_stream_cmd(cmd)

                # Flush initial samples
                for _ in range(10):
                    self.streamer.recv(self.recv_buffer, self.metadata)

            elif self.type in ('hackrf', 'rtl'):
                drv = 'hackrf' if self.type == 'hackrf' else 'rtlsdr'
                args = f"driver={drv},serial={self.serial}"
                
                self.sdr = SoapySDR.Device(args)

                self.sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
                self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
                self.sdr.setGain(SOAPY_SDR_RX, 0, gain)

                self.streamer = self.sdr.setupStream(
                    SOAPY_SDR_RX, SOAPY_SDR_CF32, [0]
                )

                self.sdr.activateStream(self.streamer, flags=0, timeNs=0)

                # Flush initial samples
                for _ in range(10):
                    sr = self.sdr.readStream(self.streamer, [self.recv_buffer], fft_size)
                    if sr.ret != fft_size:
                        raise RuntimeError(f"Failed to read stream, return code: {sr.ret}")

            else:
                raise ValueError(f"Unsupported SDR type: {self.type}")

            self.running = True
            self.acq_timer.start()
            
        except Exception as e:
            self.error_signal.emit(f"SDR setup failed: {str(e)}")

    @pyqtSlot()
    def acquire_data(self):
        if not self.running or self.sdr is None:
            return
            
        try:
            if self.type == 'usrp':
                self.streamer.recv(self.recv_buffer, self.metadata)
                samples = self.recv_buffer[0]
            else:
                # For SoapySDR devices, handle read errors more gracefully
                max_retries = 3
                for attempt in range(max_retries):
                    sr = self.sdr.readStream(self.streamer, [self.recv_buffer], fft_size, timeoutUs=100000)  # 100ms timeout
                    
                    if sr.ret == fft_size:
                        samples = self.recv_buffer[0]
                        break
                    elif sr.ret == -4:  # Timeout
                        if attempt < max_retries - 1:
                            continue
                        else:
                            # Skip this frame on persistent timeout
                            return
                    elif sr.ret < 0:
                        # Other error codes
                        if attempt < max_retries - 1:
                            continue
                        else:
                            print(f"Persistent stream error: {sr.ret}")
                            return
                    else:
                        # Partial read - use what we got if it's reasonable
                        if sr.ret > fft_size // 2:
                            # Pad with zeros if we got at least half the samples
                            padded_buffer = np.zeros(fft_size, dtype=np.complex64)
                            padded_buffer[:sr.ret] = self.recv_buffer[0][:sr.ret]
                            samples = padded_buffer
                            break
                        else:
                            # Too few samples, retry
                            if attempt < max_retries - 1:
                                continue
                            else:
                                return

            # Emit time domain data
            time_samples = min(len(samples), time_plot_samples)
            self.time_plot_update.emit(samples[:time_samples])

            # Calculate and emit frequency domain data
            if len(samples) >= fft_size:
                PSD = 10.0 * np.log10(
                    np.abs(np.fft.fftshift(np.fft.fft(samples[:fft_size])))**2 / fft_size + 1e-12
                )
                self.PSD_avg = 0.95 * self.PSD_avg + 0.05 * PSD  # Slower averaging for stability
                self.freq_plot_update.emit(self.PSD_avg.copy())

                # Update waterfall
                self.spectrogram = np.roll(self.spectrogram, 1, axis=1)
                self.spectrogram[:, 0] = PSD
                self.waterfall_plot_update.emit(self.spectrogram.copy())
            
        except Exception as e:
            self.error_signal.emit(f"Data acquisition error: {str(e)}")

    @pyqtSlot(int)
    def update_freq(self, val):
        if not self.sdr:
            return
            
        new_hz = val * 1e3
        try:
            if self.type == 'usrp':
                self.sdr.set_rx_freq(uhd.libpyuhd.types.tune_request(new_hz), 0)
            else:
                self.sdr.setFrequency(SOAPY_SDR_RX, 0, new_hz)
        except Exception as e:
            self.error_signal.emit(f"Frequency update failed: {str(e)}")

    @pyqtSlot(int)
    def update_gain(self, val):
        if not self.sdr:
            return
            
        try:
            if self.type == 'usrp':
                self.sdr.set_rx_gain(val, 0)
            else:
                self.sdr.setGain(SOAPY_SDR_RX, 0, val)
        except Exception as e:
            self.error_signal.emit(f"Gain update failed: {str(e)}")

    @pyqtSlot(int)
    def update_sample_rate(self, idx):
        if not self.sdr or idx >= len(sample_rates):
            return
            
        new_rate = int(sample_rates[idx] * 1e6)
        try:
            if self.type == 'usrp':
                self.sdr.set_rx_rate(new_rate, 0)
            else:
                self.sdr.setSampleRate(SOAPY_SDR_RX, 0, new_rate)
                
            global sample_rate
            sample_rate = new_rate
            
        except Exception as e:
            self.error_signal.emit(f"Sample rate update failed: {str(e)}")

    @pyqtSlot()
    def stop_acquisition(self):
        """Thread-safe cleanup method to be called from worker thread"""
        self.running = False
        
        # Stop timer in the same thread it was created
        if self.acq_timer:
            self.acq_timer.stop()
            self.acq_timer = None
        
        # Clean up SDR resources
        if self.streamer is not None:
            try:
                if self.type == 'usrp':
                    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                    self.streamer.issue_stream_cmd(stop_cmd)
                else:
                    self.sdr.deactivateStream(self.streamer)
                    self.sdr.closeStream(self.streamer)
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.streamer = None
        
        # Clean up SDR device
        if self.sdr is not None:
            try:
                self.sdr = None
            except Exception as e:
                print(f"Error cleaning up SDR: {e}")
        
        # Signal that cleanup is complete
        self.cleanup_finished.emit()

class MainWindow(QMainWindow):
    # Add signal to request worker cleanup
    request_cleanup = pyqtSignal()

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.current_center_freq = center_freq
        self.cleanup_complete = False
        
        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000))
        layout = QGridLayout()
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Time domain plot
        self.time_plot = pg.PlotWidget(
            labels={'left': 'Amplitude', 'bottom': 'Time [μs]'}
        )
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.curve_i = self.time_plot.plot(pen='r', name='I')
        self.curve_q = self.time_plot.plot(pen='b', name='Q')
        layout.addWidget(self.time_plot, 0, 0)

        # Frequency domain plot
        self.freq_plot = pg.PlotWidget(
            labels={'left': 'PSD (dB)', 'bottom': 'Freq (MHz)'}
        )
        self.curve_psd = self.freq_plot.plot(pen='g')
        self.freq_plot.setXRange(
            center_freq/1e6 - sample_rate/2e6,
            center_freq/1e6 + sample_rate/2e6
        )
        self.freq_plot.setYRange(-60, 10)
        layout.addWidget(self.freq_plot, 1, 0)

        # Waterfall plot
        wf_box = QHBoxLayout()
        layout.addLayout(wf_box, 2, 0)
        self.waterfall = pg.PlotWidget(
            labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'}
        )
        self.imageitem = pg.ImageItem(axisOrder='col-major')
        self.waterfall.addItem(self.imageitem)
        wf_box.addWidget(self.waterfall)
        self.hist = pg.HistogramLUTWidget()
        self.hist.setImageItem(self.imageitem)
        self.hist.item.gradient.loadPreset('viridis')
        wf_box.addWidget(self.hist)

        # Frequency slider with debouncing
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(0, int(6e6))
        freq_slider.setValue(int(center_freq/1e3))
        layout.addWidget(freq_slider, 3, 0)
        freq_label = QLabel(f"Frequency [MHz]: {center_freq/1e6:.3f}")
        layout.addWidget(freq_label, 3, 1)
        
        # Debounce timer for frequency updates (created in GUI thread)
        self.freq_timer = QTimer(self)
        self.freq_timer.setSingleShot(True)
        self.freq_timer.setInterval(100)  # 100ms debounce
        
        def on_freq_moved(val):
            freq_label.setText(f"Frequency [MHz]: {val/1e3:.3f}")
            self.current_center_freq = val * 1e3
            self.freq_timer.stop()
            try:
                self.freq_timer.timeout.disconnect()  # Clear previous connections
            except TypeError:
                pass  # No connections to disconnect
            self.freq_timer.timeout.connect(lambda: worker.update_freq(val))
            self.freq_timer.start()
            
        freq_slider.sliderMoved.connect(on_freq_moved)

        # Gain slider with debouncing
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 73)
        gain_slider.setValue(gain)
        layout.addWidget(gain_slider, 4, 0)
        gain_label = QLabel(f"Gain: {gain}")
        layout.addWidget(gain_label, 4, 1)
        
        # Debounce timer for gain updates (created in GUI thread)
        self.gain_timer = QTimer(self)
        self.gain_timer.setSingleShot(True)
        self.gain_timer.setInterval(100)  # 100ms debounce
        
        def on_gain_moved(val):
            gain_label.setText(f"Gain: {val}")
            self.gain_timer.stop()
            try:
                self.gain_timer.timeout.disconnect()  # Clear previous connections
            except TypeError:
                pass  # No connections to disconnect
            self.gain_timer.timeout.connect(lambda: worker.update_gain(val))
            self.gain_timer.start()
            
        gain_slider.sliderMoved.connect(on_gain_moved)

        # Sample rate combo box
        sample_combo = QComboBox()
        sample_combo.addItems([f"{x} MHz" for x in sample_rates])
        sample_combo.setCurrentIndex(0)
        sample_combo.currentIndexChanged.connect(worker.update_sample_rate)
        layout.addWidget(sample_combo, 5, 0)
        sample_label = QLabel(f"Sample Rate: {sample_rates[0]} MHz")
        layout.addWidget(sample_label, 5, 1)
        sample_combo.currentIndexChanged.connect(
            lambda i: sample_label.setText(f"Sample Rate: {sample_rates[i]} MHz")
        )

        # Connect worker signals
        worker.time_plot_update.connect(self._on_time)
        worker.freq_plot_update.connect(self._on_psd)
        worker.waterfall_plot_update.connect(self._on_wf)
        worker.error_signal.connect(self._on_error)
        worker.cleanup_finished.connect(self._on_cleanup_finished)
        
        # Connect cleanup request signal
        self.request_cleanup.connect(worker.stop_acquisition)

    def _on_time(self, data):
        time_axis = np.arange(len(data)) / (sample_rate / 1e6)  # microseconds
        self.curve_i.setData(time_axis, data.real)
        self.curve_q.setData(time_axis, data.imag)

    def _on_psd(self, psd):
        freqs = np.linspace(
            self.current_center_freq - sample_rate/2,
            self.current_center_freq + sample_rate/2,
            fft_size
        ) / 1e6
        self.curve_psd.setData(freqs, psd)
        
        # Update frequency plot range
        self.freq_plot.setXRange(
            self.current_center_freq/1e6 - sample_rate/2e6,
            self.current_center_freq/1e6 + sample_rate/2e6
        )

    def _on_wf(self, spec):
        self.imageitem.setImage(spec, autoLevels=False)
        
        # Calculate reasonable levels for display
        valid_data = spec[spec > -np.inf]
        if len(valid_data) > 0:
            p5, p95 = np.percentile(valid_data, [5, 95])
            self.imageitem.setLevels((p5, p95))
            self.hist.setLevels(p5, p95)

    def _on_error(self, error_msg):
        print(f"Error: {error_msg}")
    
    @pyqtSlot()
    def _on_cleanup_finished(self):
        """Slot called when worker has finished cleanup"""
        self.cleanup_complete = True

    def closeEvent(self, event):
        """Handle window close event with proper cleanup"""
        if not self.cleanup_complete:
            # Request cleanup from worker thread
            self.request_cleanup.emit()
            
            # Give worker some time to clean up
            from PyQt6.QtCore import QEventLoop, QTimer
            loop = QEventLoop()
            cleanup_timer = QTimer()
            cleanup_timer.setSingleShot(True)
            cleanup_timer.setInterval(1000)  # 1 second timeout
            
            # Connect signals to exit the event loop
            self.worker.cleanup_finished.connect(loop.quit)
            cleanup_timer.timeout.connect(loop.quit)
            
            cleanup_timer.start()
            loop.exec()  # Wait for cleanup or timeout
            
        event.accept()

def main(serial, type_, connected_flag):
    if not bool(int(connected_flag)):
        print("SDR not connected, exiting...")
        return

    global sample_rate
    global sample_rates

    if type_.lower() == 'rtl':
        sample_rates = [2.4, 2.048, 1.024]
        sample_rate  = sample_rates[0] * 1e6
    else:
        sample_rates = [10, 5, 2, 1, 0.5]
        sample_rate  = 8e6

    app = QApplication([])
    
    # Create worker and thread
    worker = SDRWorker(serial, type_)
    thread = QThread()
    worker.moveToThread(thread)
    
    # Connect thread lifecycle
    thread.started.connect(worker.setup_sdr)
    
    # Create and show main window
    win = MainWindow(worker)
    win.show()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Start the worker thread
    thread.start()
    
    # Run the application
    try:
        app.exec()
    finally:
        # Ensure proper cleanup
        if not win.cleanup_complete:
            win.request_cleanup.emit()
            thread.quit()
            thread.wait(3000)  # Wait up to 3 seconds for thread to finish
        else:
            thread.quit()
            thread.wait()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])