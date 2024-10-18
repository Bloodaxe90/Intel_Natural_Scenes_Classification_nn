import os
from torchvision import transforms
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
import zipfile

from scr.custom_datasets import CustomDataset


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


def expands_zip(zip_path: Path, extract_path: Path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def copy_images(image_dir: Path, dest_dir: Path, file_type: str, num_classes: int):
    for cls in [d for i, d in enumerate(image_dir.iterdir()) if i < num_classes and d.is_dir()]:
        class_dir = dest_dir / cls.name
        os.makedirs(class_dir, exist_ok=True)
        for image in cls.glob(f"*.{file_type}"):
            if image.is_file() and not os.path.exists(class_dir / image.name):
                shutil.copy2(image, class_dir)


def get_smaller_set(root_dir: Path, file_type: str, num_classes: int) -> Path:
  new_root_dir = Path("/Users/eric/PycharmProjects/Intel_Natural_Scenes_Classification_nn/small_data")
  train_dir = new_root_dir / "train"
  test_dir = new_root_dir / "test"
  prediction_dir = new_root_dir / "pred"

  copy_images(root_dir / "seg_train" / "seg_train", train_dir, file_type, num_classes)
  copy_images((root_dir / "seg_test" / "seg_test"), test_dir, file_type, num_classes)

  for image in (root_dir / "seg_pred" / "seg_pred").glob(f"*.{file_type}"):
    os.makedirs(prediction_dir, exist_ok=True)
    if not os.path.exists(prediction_dir / image.name):
      shutil.copy2(image, prediction_dir)

  return new_root_dir


