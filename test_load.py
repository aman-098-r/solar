import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import traceback

try:
    print("Importing TF...")
    import tensorflow as tf
    print("TF Imported.")
    from pathlib import Path
    import pickle
    
    p = Path("C:/Users/ommk0/Downloads/solar/python/solar_model.pkl")
    print("Loading model from", p)
    if p.exists():
        with open(p, "rb") as f:
            mod = pickle.load(f)
        print("Model loaded successfully")
    else:
        print("No model found!")
except Exception as e:
    print("Exception occurred!")
    traceback.print_exc()
