import pickle


class ModelTrainingLogger:
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_report': [],
            'train_confusion_matrix': [],
            'val_loss': [],
            'val_report': [],
            'val_confusion_matrix': [],
            'learning_rate': [],
            #'weights': [],
            #'gradients': [],
        }

    def log_epoch(self, epoch, train_loss, train_report, val_loss, val_report, learning_rate, train_confusion_matrix, val_confusion_matrix):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_report'].append(train_report)
        self.history['val_loss'].append(val_loss)
        self.history['val_report'].append(val_report)
        self.history['learning_rate'].append(learning_rate)
        self.history['train_confusion_matrix'].append(train_confusion_matrix)
        self.history['val_confusion_matrix'].append(val_confusion_matrix)

        #self.history['gradients'].append(gradients)
        #self.history['weights'].append(weights)

    def get_history(self):
        return self.history

    def get_values_from_reports(self, report_key, value):
        reports = self.history[report_key]

        return [report[value] for report in reports[:]]

    def get_values_from_reports_with_inside_key(self, report_key, inside_report_key, value):
        reports = self.history[report_key]

        return [report[inside_report_key][value] for report in reports[:]]

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)

        print(f"Training logs saved successfully at : {filename}\n")

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.history = pickle.load(f)

        print(f"Training logs loaded successfully from : {filename}\n")
