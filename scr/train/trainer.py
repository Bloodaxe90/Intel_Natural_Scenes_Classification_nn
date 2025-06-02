import os
from pathlib import Path

import torch.cuda
from torchinfo import summary
from torchvision import transforms
from timeit import default_timer as timer
from scr import utils, models
from scr.data_setup.dataloader_setup import create_dataloaders
from scr.engine.train import train
from scr.utils.early_stopping import EarlyStopping
from scr.utils.save_load import save_results
from scr.utils.other import set_seed, get_device


class Trainer:

    def __init__(self, epochs: int,
                 batch_size: int,
                 hidden_layers: int,
                 neurons_per_hidden_layer: list[int],
                 leaning_rate: float,
                 model_name: str = ""
                 ):
        set_seed(42)

        self.device = get_device()
        self.MODEL_NAME = model_name
        print(f"Device: {self.device}")

        #Hyper Parameters
        self.EPOCHS: int = epochs
        self.BATCH_SIZE: int = batch_size
        self.HIDDEN_LAYERS: int = hidden_layers
        self.NEURONS_PER_HIDDEN_LAYER: list[int] = neurons_per_hidden_layer
        self.LEARNING_RATE: float = leaning_rate
        self.image_size: tuple = (128, 128)


        self.WORKERS = os.cpu_count()
        print(f"Number of workers: {self.WORKERS}")

        self.train_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        root_path = Path(
            f"{os.path.dirname(os.getcwd())}/data")
        train_path = root_path / "seg_train/seg_train"
        test_path = root_path / "seg_test/seg_test"

        self.train_dataloader, self.test_dataloader, self.classes = create_dataloaders(train_path= train_path,
                                                                                   test_path=test_path,
                                                                                   train_transform= self.train_transform,
                                                                                   test_transform=self.test_transform,
                                                                                   batch_size=self.BATCH_SIZE,
                                                                                   num_workers=self.WORKERS)

        self.model_0: models.cnn_model.CNNModel = models.cnn_model.CNNModel(input_neurons= 3,
                                           num_hidden_layers= self.HIDDEN_LAYERS,
                                           neurons_per_hidden_layer= self.NEURONS_PER_HIDDEN_LAYER,
                                           output_neurons= len(self.classes),
                                           output_block_divisor= 4,
                                           image_size= self.image_size).to(self.device)

        summary(self.model_0, input_size= (32, 3, self.image_size[0], self.image_size[1]))

        self.LOSS_FN = torch.nn.CrossEntropyLoss()
        self.OPTIMIZER = torch.optim.Adam(params=self.model_0.parameters(), lr=self.LEARNING_RATE)


    def train(self):
        print("Training begun")

        start_time = timer()
        results = train(model=self.model_0,
                               train_dataloader=self.train_dataloader,
                               test_dataloader=self.test_dataloader,
                               optimizer=self.OPTIMIZER,
                               loss_fn=self.LOSS_FN,
                               epochs=self.EPOCHS,
                               device=self.device,
                               early_stopping= EarlyStopping(0.01, 5))

        print(f"Training finished | Runtime: {timer() - start_time}")
        print(results)

        if self.MODEL_NAME != "":
            utils.save_load.save_model(self.model_0,self.MODEL_NAME)
        save_results(results, self.MODEL_NAME)
