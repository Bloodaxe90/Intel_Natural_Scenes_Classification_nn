import os
import shutil
from pathlib import Path
import zipfile

def load_init_files(smaller_set_num_classes: int  = -1):
    root_path = Path(
        f"{os.path.dirname(os.path.dirname(os.getcwd()))}/IntelImageClassification.zip")
    extract_path = Path(f"{os.path.dirname(os.path.dirname(os.getcwd()))}/data")

    expands_zip(root_path, extract_path)
    if smaller_set_num_classes > 0:
        get_smaller_set(extract_path, "jpg", smaller_set_num_classes)

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
  new_root_dir = Path(f"{os.path.dirname(os.path.dirname(os.getcwd()))}/small_data")
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

load_init_files()
