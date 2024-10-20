import torch
from torch import nn
from torch.utils.data import DataLoader

from scr import custom_datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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