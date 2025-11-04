# my_ml_lib/utils/io_utils.py
import os
import pickle
from datetime import datetime

def save_model(model, model_name="model", folder="saved_models"):
    """
    Saves a trained model to disk as a pickle (.pkl).
    File name format: saved_models/{model_name}_YYYYMMDD_HHMMSS.pkl
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{model_name}_{timestamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to: {path}")
    return path

def load_model(path):
    """
    Loads a previously saved pickle model from disk.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from: {path}")
    return model
