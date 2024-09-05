import matplotlib.pyplot as plt
from Data import dataset
import pickle
import os
import seaborn as sns
import numpy as np


def load_datasets(train_birds_sample_dict_save, test_birds_sample_dict_save):
    print("=========================== Loading datasets...")
    # Use the pre-processed data in the datasets split in train and test
    train_dataset = dataset.BirdDataset(train_birds_sample_dict_save)
    test_dataset = dataset.BirdDataset(test_birds_sample_dict_save)
    print("=========================== Datasets loaded.\n")

    return train_dataset, test_dataset


def plot_training_results(logger):
    # First plot the 5 main metrics
    fig, axis = plt.subplots(3, 2, figsize=(18, 12))

    plot_loss_records(logger, axis[0, 0])
    plot_metric_records(logger, 'accuracy', 'Train and Validation Accuracy', 'Accuracy', axis[0, 1])
    plot_metric_records(logger, 'precision', 'Train and Validation Precision', 'Precision', axis[1, 0], 'macro avg', 'precision')
    plot_metric_records(logger, 'recall', 'Train and Validation Recall', 'Recall', axis[1, 1], 'macro avg', 'recall')
    plot_metric_records(logger, 'f1-score', 'Train and Validation F1-score', 'F1-score', axis[2, 0], 'macro avg', 'f1-score')

    plt.tight_layout()
    plt.show()

    # Then plot the confusion matrices for train and validation
    plot_confusion_matrix(logger)


def plot_loss_records(logger, axis):
    history = logger.get_history()

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = history['epoch']

    # Create the plot
    axis.plot(epochs, train_loss, marker='o', linestyle='-', color='b', label='Train loss')
    axis.plot(epochs, val_loss, marker='x', linestyle='-', color='r', label='Validation loss')
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Loss values')
    axis.set_title('Train and Validation Loss Values Through Epochs')
    axis.legend()
    axis.grid(True)


def plot_metric_records(logger, metric_name, title, ylabel, axis, report_key=None, report_subkey=None):
    try:
        history = logger.get_history()

        if report_subkey:
            train_values = logger.get_values_from_reports_with_inside_key("train_report", report_key, report_subkey)
            val_values = logger.get_values_from_reports_with_inside_key("val_report", report_key, report_subkey)
        else:
            train_values = logger.get_values_from_reports("train_report", metric_name)
            val_values = logger.get_values_from_reports("val_report", metric_name)

        epochs = history['epoch']

        axis.plot(epochs, train_values, marker='o', linestyle='-', color='b', label=f'Train {metric_name}')
        axis.plot(epochs, val_values, marker='x', linestyle='-', color='r', label=f'Validation {metric_name}')
        axis.set_xlabel('Epochs')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
        axis.grid(True)
    except KeyError as e:
        print(f"Error: {e} not found in the logger's history.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")


def plot_confusion_matrix(logger):
    history = logger.get_history()

    train_confusion_matrix = history['train_confusion_matrix']
    val_confusion_matrix = history['val_confusion_matrix']

    train_mean_confusion_matrix = np.sum(train_confusion_matrix, axis=0) / len(train_confusion_matrix)
    val_mean_confusion_matrix = np.sum(val_confusion_matrix, axis=0) / len(val_confusion_matrix)

    n_birds = len(train_mean_confusion_matrix[0])
    birds_indexes = list(range(1, n_birds + 1))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Adjust size depending on your needs

    # Plot the first confusion matrix
    sns.heatmap(train_mean_confusion_matrix, annot=False, cmap=plt.cm.Blues, xticklabels=birds_indexes, yticklabels=birds_indexes, ax=axes[0])
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='y', rotation=0)

    # Plot the second confusion matrix
    sns.heatmap(val_mean_confusion_matrix, annot=False, cmap=plt.cm.Blues, xticklabels=birds_indexes, yticklabels=birds_indexes, ax=axes[1])
    axes[1].set_title('Validation Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='y', rotation=0)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def save_model(model_save_path, model):
    try:
        file_name = os.path.join(model_save_path, f"{model.name}.pkl")
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved successfully to: {file_name}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model_save_path, model_name):
    try:
        file_name = os.path.join(model_save_path, f"{model_name}.pkl")
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        print(f"Model {model_name} loaded successfully from: {file_name}")
        return model

    except FileNotFoundError:
        print(f"Model file {file_name} not found.")
    except Exception as e:
        print(f"Error loading model: {e}")
