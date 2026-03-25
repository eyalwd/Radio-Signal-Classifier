import dsp_utils

# --- THE MAIN ORCHESTRATOR ---
def main():
    source = dsp_utils.SDRSource(freq=1090e6, sample_rate=2.048e6)
    processor = dsp_utils.DSPProcessor()
    factory = dsp_utils.FeatureFactory()
    classifier = dsp_utils.RadioClassifier("my_model.pkl")

    try:
        while True:
            raw_data = source.capture()
            psd_data = processor.to_frequency_domain(raw_data)
            features = factory.get_vector(raw_data, psd_data)
            
            result = classifier.identify(features)
            print(f"Current Signal: {result}")
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()