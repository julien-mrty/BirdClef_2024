import numpy as np


class TrainingLogger:
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'gradients': [],
            'learning_rate': [],
            'weights': [],
        }

    def log_epoch(self, epoch, train_loss, train_accuracy, val_loss, val_accuracy, gradients, learning_rate, weights):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_accuracy'].append(train_accuracy)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        self.history['gradients'].append(gradients)
        self.history['learning_rate'].append(learning_rate)
        self.history['weights'].append(weights)

    def get_history(self):
        return self.history

    def save_to_file(self, filename='training_log.npy'):
        np.save(filename, self.history)

    def load_from_file(self, filename='training_log.npy'):
        self.history = np.load(filename, allow_pickle=True).item()
