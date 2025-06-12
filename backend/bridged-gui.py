#!/usr/bin/env python3
import warnings
# —————————————————————————————————————————————————————————————
# SUPPRESS NumPy’s thread-unsafe warning-filter path
warnings.filterwarnings('ignore', category=RuntimeWarning)
# —————————————————————————————————————————————————————————————

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

# —————————————————————————————————————————————————————————————
# Global defaults
fft_size          = 4096
num_rows          = 200
center_freq       = 750e6
sample_rates = []
sample_rate = None
time_plot_samples = 500
gain              = 50
# —————————————————————————————————————————————————————————————

class SDRWorker(QObject):
    # Signals to drive GUI updates
    time_plot_update      = pyqtSignal(np.ndarray)
    freq_plot_update      = pyqtSignal(np.ndarray)
    waterfall_plot_update = pyqtSignal(np.ndarray)
    end_of_run            = pyqtSignal()

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

    @pyqtSlot()
    def setup_sdr(self):
        if self.type == 'usrp':
            # — USRP via UHD —
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

            # start continuous streaming
            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            self.streamer.issue_stream_cmd(cmd)

            # flush initial samples
            for _ in range(10):
                self.streamer.recv(self.recv_buffer, self.metadata)

        elif self.type in ('hackrf', 'rtl'):
            drv = 'hackrf' if self.type == 'hackrf' else 'rtlsdr'
            args = f"driver={drv},serial={self.serial}"
            try:
                self.sdr = SoapySDR.Device(args)
            except RuntimeError as e:
                print(f"Failed to open {drv}: {e}")
                self.end_of_run.emit()
                return

            # Check if the device is initialized
            if self.sdr is None:
                print(f"Failed to initialize {drv} device.")
                self.end_of_run.emit()
                return

            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
            self.sdr.setGain(SOAPY_SDR_RX, 0, gain)

            self.streamer = self.sdr.setupStream(
                SOAPY_SDR_RX, SOAPY_SDR_CF32, [0]
            )

            if self.streamer is None:
                print(f"Failed to set up stream for {drv}.")
                self.end_of_run.emit()
                return

            self.sdr.activateStream(self.streamer, flags=0, timeNs=0)

            # flush initial samples
            for _ in range(10):
                sr = self.sdr.readStream(self.streamer, [self.recv_buffer], fft_size)
                if sr.ret != fft_size:
                    print(f"Failed to read stream, return code: {sr.ret}")
                    self.end_of_run.emit()
                    return

        else:
            raise ValueError(f"Unsupported SDR type: {self.type}")

        QTimer.singleShot(0, self.run)


    def run(self):
        """Grab one buffer, compute FFT & waterfall, emit signals, schedule next."""
        if self.type == 'usrp':
            self.streamer.recv(self.recv_buffer, self.metadata)
            samples = self.recv_buffer[0]
        else:
            # hackrf / rtl
            sr = self.sdr.readStream(self.streamer, [self.recv_buffer], fft_size)
            samples = self.recv_buffer[0]

        # 1) Time-domain slice
        self.time_plot_update.emit(samples[:time_plot_samples])

        # 2) Compute PSD & EMA average
        PSD = 10.0 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / fft_size
        )
        self.PSD_avg = 0.99 * self.PSD_avg + 0.01 * PSD
        self.freq_plot_update.emit(self.PSD_avg)

        # 3) Update waterfall buffer
        self.spectrogram[:]    = np.roll(self.spectrogram, 1, axis=1)
        self.spectrogram[:, 0] = PSD
        self.waterfall_plot_update.emit(self.spectrogram)

        # 4) Loop
        self.end_of_run.emit()

    # --- the rest of your slider-handling slots, exactly as before ---
    @pyqtSlot(int)
    def update_freq(self, val):
        new_hz = val * 1e3
        if self.type == 'usrp':
            self.sdr.set_rx_freq(uhd.libpyuhd.types.tune_request(new_hz), 0)
        else:
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, new_hz)
        # flush a few samples
        for _ in range(3):
            self.run()

    @pyqtSlot(int)
    def update_gain(self, val):
        if self.type == 'usrp':
            self.sdr.set_rx_gain(val, 0)
        else:
            self.sdr.setGain(SOAPY_SDR_RX, 0, val)
        for _ in range(3):
            self.run()

    @pyqtSlot(int)
    def update_sample_rate(self, idx):
        sr = sample_rates[idx] * 1e6
        if self.type == 'usrp':
            self.sdr.set_rx_rate(sr, 0)
        else:
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, sr)
        for _ in range(3):
            self.run()


class MainWindow(QMainWindow):
    def __init__(self, worker):
        super().__init__()
        self.setWindowTitle("The PySDR Spectrum Analyzer")
        self.setFixedSize(QSize(1500, 1000))
        layout = QGridLayout()
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Time plot
        self.time_plot = pg.PlotWidget(
            labels={'left': 'Amplitude', 'bottom': 'Time [μs]'}
        )
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.curve_i = self.time_plot.plot()
        self.curve_q = self.time_plot.plot()
        layout.addWidget(self.time_plot, 0, 0)

        # PSD plot
        self.freq_plot = pg.PlotWidget(
            labels={'left': 'PSD (dB)', 'bottom': 'Freq (MHz)'}
        )
        self.curve_psd = self.freq_plot.plot()
        self.freq_plot.setXRange(
            center_freq/1e6 - sample_rate/2e6,
            center_freq/1e6 + sample_rate/2e6
        )
        self.freq_plot.setYRange(-30, 20)
        layout.addWidget(self.freq_plot, 1, 0)

        # Waterfall + colorbar
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

        # Controls
        freq_slider = QSlider(Qt.Orientation.Horizontal)
        freq_slider.setRange(0, int(6e6))
        freq_slider.setValue(int(center_freq/1e3))
        freq_slider.sliderMoved.connect(worker.update_freq)
        layout.addWidget(freq_slider, 3, 0)
        freq_label = QLabel(f"Frequency [MHz]: {center_freq/1e6}")
        layout.addWidget(freq_label, 3, 1)
        freq_slider.sliderMoved.connect(
            lambda v: freq_label.setText(f"Frequency [MHz]: {v/1e3}")
        )

        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 73)
        gain_slider.setValue(gain)
        gain_slider.sliderMoved.connect(worker.update_gain)
        layout.addWidget(gain_slider, 4, 0)
        gain_label = QLabel(f"Gain: {gain}")
        layout.addWidget(gain_label, 4, 1)
        gain_slider.sliderMoved.connect(
            lambda v: gain_label.setText(f"Gain: {v}")
        )

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

        # Connect plotting signals
        worker.time_plot_update.connect(self._on_time)
        worker.freq_plot_update.connect(self._on_psd)
        worker.waterfall_plot_update.connect(self._on_wf)
        worker.end_of_run.connect(lambda: QTimer.singleShot(0, worker.run))

    def _on_time(self, data):
        self.curve_i.setData(data.real)
        self.curve_q.setData(data.imag)

    def _on_psd(self, psd):
        freqs = np.linspace(
            center_freq - sample_rate/2,
            center_freq + sample_rate/2,
            fft_size
        ) / 1e6
        self.curve_psd.setData(freqs, psd)

    def _on_wf(self, spec):
        self.imageitem.setImage(spec, autoLevels=False)
        sigma = np.std(spec); mean = np.mean(spec)
        lo, hi = mean - 2*sigma, mean + 2*sigma
        self.imageitem.setLevels((lo, hi))
        self.hist.setLevels(lo, hi)


def main(serial, type_, connected_flag):
    if not bool(int(connected_flag)):
        print("SDR not connected, exiting...")
        return

    global sample_rate
    global sample_rates

    if type_.lower() == 'rtl':
        sample_rates = [2.4, 2.0, 1.0, 0.5]
    else:
        sample_rates = [10, 5, 2, 1, 0.5]

    if type_.lower() == 'rtl':
        sample_rate = 2.4e6
    else:
        sample_rate = 8e6

    app = QApplication([])
    worker = SDRWorker(serial, type_)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.setup_sdr)

    win = MainWindow(worker)
    win.show()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    thread.start()
    app.exec()

    # Cleanly stop stream on exit
    if hasattr(worker, 'streamer') and worker.streamer is not None:
        if worker.type == 'usrp':
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            worker.streamer.issue_stream_cmd(stop_cmd)
        else:
            try:
                worker.sdr.deactivateStream(worker.streamer)
                worker.sdr.closeStream(worker.streamer)
            except Exception:
                pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])