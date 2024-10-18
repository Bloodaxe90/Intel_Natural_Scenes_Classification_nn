import torch
import numpy as np
from PIL.Image import Image
from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, root_path: Path, transform: transforms = None):
    super().__init__()
    self.data_path = root_path
    self.transform = transform
    self.images = list(self.data_path.glob("*/*.jpg"))

    self.labels = [img.parent.name for img in self.images]
    self.classes = list(np.unique(self.labels))
    self.class_dict = {i : cls for i, cls in enumerate(self.classes)}

  def get_pil_image(self, index) -> Image.Image:
    return Image.open(self.images[index])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index) -> tuple[torch.Tensor, int]:
    image = self.get_pil_image(index)
    if self.transform is not None:
      image = self.transform(image)
    return image, self.classes.index(self.labels[index])