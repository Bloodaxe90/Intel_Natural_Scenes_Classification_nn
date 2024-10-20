from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader

from scr.custom_datasets.custom_dataset import CustomDataset

def create_dataloaders(train_path: Path,
                       test_path: Path,
                       train_transform: transforms,
                       test_transform: transforms,
                       batch_size: int,
                       num_workers: int
                       ) -> tuple[DataLoader, DataLoader, list]:

    train_dataset = CustomDataset(train_path, train_transform)
    test_dataset = CustomDataset(test_path, test_transform)

    class_names = train_dataset.classes

    train_dataloader = DataLoader(dataset= train_dataset,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= num_workers,
                                  pin_memory= True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory= True)

    return train_dataloader, test_dataloader, class_names


