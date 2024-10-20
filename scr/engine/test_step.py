import torch
from torch import nn
from torch.utils.data import DataLoader

def test_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              device: str):
  model.eval()

  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    for batch in dataloader:
      X, y = batch
      X, y = X.to(device), y.to(device)

      test_pred = model(X)

      loss = loss_fn(test_pred, y)
      test_loss += loss.item()

      test_pred_class = torch.argmax(test_pred, dim=1)
      test_acc += (test_pred_class == y).sum().item()/len(test_pred)

  test_loss /= len(dataloader)
  test_acc /= len(dataloader)
  return test_loss, test_acc