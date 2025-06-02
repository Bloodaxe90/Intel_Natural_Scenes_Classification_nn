import torch
from pandas import DataFrame
from torch import nn
from torch.utils.data import DataLoader

from scr.engine.test_step import test_step
from scr.engine.train_step import train_step
from scr.utils.early_stopping import EarlyStopping


def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: str,
          early_stopping: EarlyStopping = None) -> DataFrame:

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


      if early_stopping is not None:
          if early_stopping(test_loss):
              print(f"Early Stopping Triggered at Epoch: {epoch}")
              break

    return results