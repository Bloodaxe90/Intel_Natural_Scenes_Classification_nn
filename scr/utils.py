import os
from pathlib import Path

import torch
from torch import nn
from scr.custom_datasets import CustomDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):

    target_path: Path = Path(target_dir)
    os.makedirs(target_path, exist_ok= True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Models name needs to end with .pth or .pt"

    model_path: Path = target_path / model_name
    print(f"Model {model_name} saved to {model_path}")
    torch.save(obj= model.state_dict(), f= model_path)

def load_model_accuracy(model, model_dir: str, dataset: CustomDataset, device: str) -> float:
    model.load_state_dict(torch.load(model_dir, map_location=device, weights_only=True))

    assert model_dir.endswith(".pth") or model_dir.endswith(".pt"), "Models name needs to end with .pth or .pt"

    model.to(device)

    correct = 0
    for i in range(len(dataset)):
        image, label = dataset[i]

        prediction = make_predictions(model, image, device)

        if prediction == label:
            correct += 1

    return correct / len(dataset)

def make_predictions(model: nn.Module,
                     image: torch.Tensor,
                     device: str) -> int:
  model.eval()

  with torch.inference_mode():
    image = image.unsqueeze(0).to(device)
    y_pred = model(image)
    y_pred_label = torch.argmax(y_pred, dim=1)

  return int(y_pred_label)




