import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_saved_file(filename, center_freq, sample_rate=2.048e6):
    # Route to the data/raw folder
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', filename))
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find the file at {filepath}")
        print("Check the spelling or make sure you are running this from inside the scripts folder.")
        return

    print(f"Loading {filename}...")
    iq_samples = np.load(filepath)
    print(f"Success! Loaded {len(iq_samples)} complex samples.")
    
    # 1. DSP: Remove DC Offset
    iq_samples = iq_samples - np.mean(iq_samples)
    
    print("Generating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # --- Plot 1: Time Domain (First 500 samples) ---
    view_window = 500 
    t = np.arange(view_window) / sample_rate
    
    ax1.plot(t * 1e6, np.real(iq_samples[:view_window]), label='In-Phase (I)', color='blue', alpha=0.8)
    ax1.plot(t * 1e6, np.imag(iq_samples[:view_window]), label='Quadrature (Q)', color='orange', alpha=0.8)
    ax1.set_title(f"Time Domain: {filename} (First {view_window} points)")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Frequency Domain (Welch's PSD) ---
    f, pxx = signal.welch(iq_samples, sample_rate, window='hann', nperseg=2048, return_onesided=False)
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    pxx_db = 10 * np.log10(pxx + 1e-12)
    
    # Convert frequencies to absolute MHz based on the center frequency it was recorded at
    f_mhz = (f + center_freq) / 1e6
    
    ax2.plot(f_mhz, pxx_db, color='purple')
    ax2.set_title("Frequency Domain: Power Spectral Density")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Power / Frequency (dB/Hz)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change these variables to match the exact file you want to look at
    TARGET_FILE = "FM_RADIO_000.npy"  # Make sure this matches a file in your data/raw folder
    RECORDED_FREQ = 91.8e6            # The frequency you were tuned to when recording it
    
    plot_saved_file(filename=TARGET_FILE, center_freq=RECORDED_FREQ)