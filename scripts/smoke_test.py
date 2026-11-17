"""Phase 1 step 2: verify the environment imports cleanly."""
import importlib

for mod in ["tensorflow", "qkeras", "tensorflow_model_optimization", "hls4ml",
            "sklearn", "pandas", "h5py", "yaml", "joblib"]:
    try:
        m = importlib.import_module(mod)
        print(f"ok   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {mod}: {e}")
