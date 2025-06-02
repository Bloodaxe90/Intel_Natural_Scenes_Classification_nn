import os
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from scr.models import cnn_model


def save_model(model: nn.Module,
               model_name: str):

    target_path = Path(f"{os.path.dirname(os.getcwd())}/saved_models")
    os.makedirs(target_path, exist_ok= True)

    if not model_name.endswith(".pth") or not model_name.endswith(".pt"):
        model_name += ".pt"

    model_path: Path = target_path / model_name
    print(f"Model {model_name} saved to {model_path}")
    torch.save(obj= model.state_dict(), f= model_path)

def load_model(model: cnn_model.CNNModel, model_name: str, device: str):
    if not model_name.endswith(".pth") or not model_name.endswith(".pt"):
        model_name += ".pt"
    model_path = Path(f"{os.path.dirname(os.getcwd())}/saved_models/{model_name}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    return model

def save_results(results: pd.DataFrame, model_name: str):
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"

    os.makedirs(log_dir, exist_ok=True)
    if ".csv" not in model_name:
        model_name += ".csv"

    results.to_csv(f"{log_dir}/{model_name}", index=False)
    print(f"Saved {model_name} in directory: {log_dir}")

def load_results(model_name: str) -> pd.DataFrame:
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"

    os.makedirs(log_dir, exist_ok=True)

    if ".csv" not in model_name:
        model_name += ".csv"
    print(f"Loaded {model_name} from directory: {log_dir}")
    return pd.read_csv(f"{log_dir}/{model_name}")