import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader


def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: str) -> DataFrame:

    results = DataFrame(columns= ["epoch", 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    for epoch in range(epochs):

      train_loss, train_acc = train_step(model= model,
                                        dataloader= train_dataloader,
                                        loss_fn= loss_fn,
                                        optimizer= optimizer,
                                        device= device)

      test_loss, test_acc = test_step(model= model,
                                      dataloader= test_dataloader,
                                      loss_fn= loss_fn,
                                      device= device)

      results.loc[len(results)] = [epoch +1, train_loss, train_acc, test_loss, test_acc]

    return results


def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str):
  model.train()

  train_loss, train_acc = 0, 0

  for batch in dataloader:
    X, y = batch
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(y_pred, dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  return train_loss, train_acc


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