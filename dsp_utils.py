import numpy as np
from rtlsdr import RtlSdr
import joblib # For loading the ML model

class SDRSource:
    def __init__(self, freq, sample_rate):
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = freq
        self.sdr.gain = 'auto'

    def capture(self, num_samples=256*1024):
        return self.sdr.read_samples(num_samples)

class DSPProcessor:
    @staticmethod
    def to_frequency_domain(iq_samples):
        # Apply Hann window and perform FFT
        window = np.hanning(len(iq_samples))
        fft_res = np.fft.fft(iq_samples * window)
        psd = np.abs(np.fft.fftshift(fft_res))**2
        return psd

class FeatureFactory:
    def get_vector(self, iq_samples, psd):
        # Calculate features (Physics-based extraction)
        amp = np.abs(iq_samples)
        stats = [np.mean(amp), np.std(amp), np.max(psd)/np.mean(psd)]
        # Return as a single row for the model
        return np.array(stats).reshape(1, -1)

class RadioClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def identify(self, feature_vector):
        return self.model.predict(feature_vector)


if __name__ == "__main__":
    pass