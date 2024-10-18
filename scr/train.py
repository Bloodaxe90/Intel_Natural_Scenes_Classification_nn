import os
from pathlib import Path

import torch.cuda
from torchvision import transforms

from scr import engine, data_setup, models
from scr.models import SmallModelCNN


def main():

    EPOCHS: int = 1
    BATCH_SIZE: int = 32
    HIDDEN_LAYERS: int = 1
    NEURONS_PER_HIDDEN_LAYER: list[int] = [16, 16]
    WORKERS = os.cpu_count()
    LEARNING_RATE: float = 0.001

    small_root_path = Path("/Users/eric/PycharmProjects/Intel_Natural_Scenes_Classification_nn/small_data")
    small_train_path = small_root_path / "train"
    small_test_path = small_root_path / "test"

    print("Is CUDA available: ", torch.cuda.is_available())
    print("Number of GPUs: ", torch.cuda.device_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    image_size: tuple[int, int] = (128, 128)
    small_train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    small_test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(train_path= small_train_path,
                                                                    test_path= small_test_path,
                                                                    train_transform= small_train_transform,
                                                                    test_transform= small_test_transform,
                                                                    batch_size= BATCH_SIZE,
                                                                    num_workers= WORKERS)

    model_0: SmallModelCNN = models.SmallModelCNN(input_neurons= 3,
                                           hidden_layers= HIDDEN_LAYERS,
                                           neurons_per_hidden_layer= NEURONS_PER_HIDDEN_LAYER,
                                           output_neurons= len(classes),
                                           image_size= image_size)

    LOSS_FN = torch.nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(params= model_0.parameters(), lr= LEARNING_RATE)

    print("training begun")
    engine.train(model = model_0,
                 train_dataloader= train_dataloader,
                 test_dataloader= test_dataloader,
                 optimizer= OPTIMIZER,
                 loss_fn= LOSS_FN,
                 epochs= EPOCHS,
                 device= device)
    print("training finished")

if __name__ == "__main__":
    main()