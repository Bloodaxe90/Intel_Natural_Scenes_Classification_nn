import os
from pathlib import Path

import torch
from torch import nn
from scr.models import cnn_model


def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):

    target_path: Path = Path(target_dir)
    os.makedirs(target_path, exist_ok= True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Models name needs to end with .pth or .pt"

    model_path: Path = target_path / model_name
    print(f"Model {model_name} saved to {model_path}")
    torch.save(obj= model.state_dict(), f= model_path)

def load_model(model: cnn_model.CNNModel, model_dir: str, device: str):
    model_path = Path(model_dir)
    model_loaded = model
    model_loaded.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model_loaded.to(device)
    return model_loaded