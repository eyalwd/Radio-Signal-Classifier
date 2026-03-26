import os
import sys
import numpy as np

# Adjust the path so the script can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hardware import SDRInterface

def build_dataset():
    # Define ground truth targets (Labels: Frequency in Hz)
    targets = {
        "FM_Radio": 91.8e6,
        "ADSB_AVIATION": 1090e6,
    }

    # Setup the path to save the data
    save_dir = os.path.join("data", "raw")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize SDR interface
    sdr = SDRInterface(sample_rate=2.4e6)
    samples_per_class = 50 # start small for testing

    try:
        for label, freq in targets.items():
            print(f"\n--- Tuning to {freq/1e6:.1f} MHz for {label} ---")

            for i in range(samples_per_class):
                # Capture raw data
                iq_data = sdr.tune_and_capture(center_frequency=freq)

                # Save a binary numpy file
                filename = os.path.join(save_dir, f"{label}_{i:03d}.npy")
                np.save(filename, iq_data)

                # Print progress on the same line
                sys.stdout.write(f"\rCaptured {i+1}/{samples_per_class} for {label}")
                sys.stdout.flush()

    except Exception as e:
                    print(f"\nError during collection: {e}")
    finally:
        sdr.close()
        print(f"\n\nHardware released. Data collection completed.")

if __name__ == "__main__":
    build_dataset()


