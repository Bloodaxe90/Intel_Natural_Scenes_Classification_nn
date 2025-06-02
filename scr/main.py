from scr.train.trainer import Trainer
from scr.utils.inference import show_confusion_matrix, \
    plot_results, get_pred_and_labels, show_predictions
from scr.utils.save_load import load_model, load_results
from sklearn.metrics import confusion_matrix


def main():
    INFERENCE: bool = True

    EPOCHS: int = 30
    BATCH_SIZE: int = 32
    HIDDEN_LAYERS: int = 2
    NEURONS_PER_HIDDEN_LAYER: list = [128, 64, 32]
    LEARNING_RATE: float = 0.0001
    PATIENCE: int = 5
    MIN_DELTA: float = 0.01
    NEW_MODEL_NAME: str = "test"

    # Specific for INFERENCE
    LOAD_MODEL_NAME: str = "model_0_train1"

    trainer = Trainer(epochs= EPOCHS,
                      batch_size= BATCH_SIZE,
                      hidden_layers= HIDDEN_LAYERS,
                      neurons_per_hidden_layer= NEURONS_PER_HIDDEN_LAYER,
                      leaning_rate= LEARNING_RATE,
                      patience= PATIENCE,
                      min_delta= MIN_DELTA,
                      model_name=NEW_MODEL_NAME)
    if not INFERENCE:
        trainer.train()
    else:
        print("INFERENCE")
        loaded_model = load_model(trainer.model_0,
                                  LOAD_MODEL_NAME,
                                  trainer.device)

        results = load_results(LOAD_MODEL_NAME)

        plot_results(results)

        show_predictions(loaded_model, trainer.test_dataloader.dataset, trainer.device)

        ypred, ytrue = get_pred_and_labels(loaded_model, trainer.test_dataloader, trainer.device)

        show_confusion_matrix(confusion_matrix(ypred, ytrue), trainer.test_dataloader.dataset)



if __name__ == "__main__":
    main()






