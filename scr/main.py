from scr.train.trainer import Trainer
from scr.utils.inference import show_predictions, get_pred_and_labels, show_confusion_matrix
from scr.utils.save_load import load_model
from sklearn.metrics import confusion_matrix


def main():
    trainer = Trainer(epochs= 30,
                      batch_size= 32,
                      hidden_layers= 2,
                      neurons_per_hidden_layer= [128, 64, 32],
                      leaning_rate= 0.0001)
    loaded_model = load_model(trainer.model_0,
                              "/shared/storage/cs/studentscratch/kkf525/PyCharm_Projects/Intel_Natural_Scenes_Classification_nn/saved_models/model_0_train1.pt",
                              trainer.device)

    ypred, ytrue = get_pred_and_labels(loaded_model, trainer.test_dataloader, trainer.device)

    count = 0
    for i in range(len(ytrue)):
        if ypred[i] == ytrue[i]:
            count += 1

    print(f"Accuracy: {count / len(ytrue)}")

    show_confusion_matrix(confusion_matrix(ypred, ytrue), trainer.test_dataloader.dataset)




if __name__ == "__main__":
    main()






