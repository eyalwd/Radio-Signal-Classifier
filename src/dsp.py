import numpy as np
from scipy import signal

def remove_dc(iq_samples):
    """Remove the DC component from the input samples."""
    return iq_samples - np.mean(iq_samples)

def normalize_amplitude(iq_samples):
    """Scales maximum absolute amplitude to 1.0 to ensure gain invariance."""
    max_value = np.max(np.abs(iq_samples))
    # Avoid division by zero
    if max_value > 0: 
        return iq_samples/max_value
    return iq_samples

def compute_psd(iq_samples, fs=2.048e6):
    """
    Computes the Power Spectral Density using Welch's method.
    Returns frequencies, PSD in dB, and raw linear PSD.
    """
    f, pxx = signal.welch(iq_samples, fs, window='hann', nperseg=1024, return_onesided=False)

    # Shift the zero frequency component to the center of the spectrum
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)

    # Convert to dB
    pxx_db = 10 * np.log10(pxx + 1e-12)  # Add small value to avoid log(0)

    return f, pxx_db, pxx