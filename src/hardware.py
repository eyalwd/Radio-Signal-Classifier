import time
import numpy as np
from rtlsdr import RtlSdr

#this calss handkes the harware transiants and the data collection
class SDRInterface:
    def __init__(self, sample_rate=2.4e6, center_freq=100e6, gain='auto'):
        """Initialize the SDR interface with the specified parameters."""
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        self.sdr.gain = gain

    def tune_and_capture(self, center_frequency, num_samples = 2 ** 18, drop_samples = 16384):
        """Tune to the specified center frequency and capture samples."""
        # Setting center frequency
        self.sdr.center_freq = center_frequency

        # Allowing time for the hardware to stabilize
        time.sleep(0.05)

        # Dump the first samples to flush the USB buffer
        _ = self.sdr.read_samples(drop_samples)

        # Capture the desired number of samples
        samples = self.sdr.read_samples(num_samples)
        return samples
    
    def close(self):
        """Close the SDR interface."""
        self.sdr.close()