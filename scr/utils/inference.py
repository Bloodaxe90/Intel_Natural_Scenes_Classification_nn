import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scr import custom_datasets
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_results(result: pd.DataFrame):

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].plot(result["epoch"], result["train_loss"], label="Train Loss")
    ax[0, 0].set_title("Train Loss")
    ax[0, 0].set_xlabel("Epochs")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].legend()

    ax[0, 1].plot(result["epoch"], result["train_acc"], label="Train Accuracy", color='green')
    ax[0, 1].set_title("Train Accuracy")
    ax[0, 1].set_xlabel("Epochs")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()

    ax[1, 0].plot(result["epoch"], result["test_loss"], label="Test Loss", color='red')
    ax[1, 0].set_title("Test Loss")
    ax[1, 0].set_xlabel("Epochs")
    ax[1, 0].set_ylabel("Loss")
    ax[1, 0].legend()

    ax[1, 1].plot(result["epoch"], result["test_acc"], label="Test Accuracy", color='purple')
    ax[1, 1].set_title("Test Accuracy")
    ax[1, 1].set_xlabel("Epochs")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].legend()

    fig.suptitle("Training and Testing Metrics Over Epochs", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

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

def get_pred_and_labels(model: nn.Module, data_loaders: DataLoader, device) -> tuple[list[int], list[int]]:
    y_pred = []
    y_true = []

    model.eval()

    with torch.inference_mode():
        for batch in data_loaders:
            X, y = batch

            X = X.to(device)
            y_pred.extend(torch.argmax(model(X), dim=1).cpu().numpy())
            y_true.extend(y.numpy())

    return y_pred, y_true

def show_confusion_matrix(conf_mat: confusion_matrix, dataset: custom_datasets):
    plt.figure(figsize=(10, 7))
    classes = list(dataset.classes)
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes)



    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def make_predictions(model: nn.Module,
                     image: torch.Tensor,
                     device: str) -> int:
  model.eval()

  with torch.inference_mode():
    image = image.unsqueeze(0).to(device)
    y_pred = model(image)
    y_pred_label = torch.argmax(y_pred, dim=1)

  return int(y_pred_label)


def show_predictions(model: nn.Module, test_dataset: custom_datasets, device, num_predictions: int = 9):
    plt.figure(figsize=(15, 15))
    for i, num in enumerate(np.random.randint(0, len(test_dataset), num_predictions)):
        image, label = test_dataset[num]

        prediction = make_predictions(model, image, device)

        plt.subplot(3, 3, i + 1)
        plt.title(f"Prediction: {test_dataset.classes[prediction]}\nActual Label: {test_dataset.classes[label]}")
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')

    plt.tight_layout()
    plt.show()