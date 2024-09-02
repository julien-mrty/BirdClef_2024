import csv
from itertools import zip_longest
import pandas as pd
import matplotlib.pyplot as plt
from Data import dataset


def load_datasets(train_birds_sample_dict_save, test_birds_sample_dict_save):
    print("=========================== Loading datasets...")
    # Use the pre-processed data in the datasets split in train and test
    train_dataset = dataset.BirdDataset(train_birds_sample_dict_save)
    test_dataset = dataset.BirdDataset(test_birds_sample_dict_save)
    print("=========================== Datasets loaded.\n")

    return train_dataset, test_dataset


def save_training_logs(my_model, save_dir):
    # Loss values
    train_loss_coordinates = my_model.train_loss_coordinates
    test_loss_coordinates = my_model.test_loss_coordinates

    file_name = save_dir + my_model.model_name + "_loss_record.h5"

    data = pd.DataFrame({
        "train_loss_coordinates": train_loss_coordinates,
        "test_loss_coordinates": test_loss_coordinates
    })

    data.to_hdf(file_name, key="df", mode="w")


def plot_loss_records(directory_training_result_save, my_model=None):
    if my_model is None:
        file_path = directory_training_result_save
    else:
        file_path = directory_training_result_save + my_model.model_name + "_loss_record.h5"

    data = pd.read_hdf(file_path, key="df")
    # Extract the coordinates from the DataFrame
    train_loss, train_epoch = zip(*data['train_loss_coordinates'])
    test_loss, test_epoch = zip(*data['test_loss_coordinates'])

    # Create the plot
    plt.figure(figsize=(12, 12))
    plt.plot(train_epoch, train_loss, marker='o', linestyle='-', color='b', label='Train loss')
    plt.plot(test_epoch, test_loss, marker='x', linestyle='-', color='r', label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss values')
    plt.title('Train and test loss values trough epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
