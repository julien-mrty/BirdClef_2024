import pandas as pd
import matplotlib.pyplot as plt
from Data import dataset
import pickle


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

    file_name = save_dir + my_model.name + "_loss_record.h5"

    data = pd.DataFrame({
        "train_loss_coordinates": train_loss_coordinates,
        "test_loss_coordinates": test_loss_coordinates
    })

    data.to_hdf(file_name, key="df", mode="w")

    print(f"Training logs saved successfully at : {save_dir}\n")


def plot_loss_records(directory_training_result_save, my_model_name=None):
    if my_model_name is None:
        file_path = directory_training_result_save
    else:
        file_path = directory_training_result_save + my_model_name + "_loss_record.h5"

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


def save_model(model_save_path, model):
    file_name = model_save_path + model.name + ".pkl"
    # Open the file in binary write mode and use pickle to dump the instance
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved successfully to : {model_save_path}\n")


def load_model(model_save_path, model_name):
    file_name = model_save_path + model_name + ".pkl"

    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    print(f"Model {model_name} loaded successfully to : \n")

    return model
