class EarlyStopping:
    def __init__(self, min_delta=0, patience=5):

        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop