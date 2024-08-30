from os.path import split

from Data import dataset
from pathlib import Path
from Data import data_pre_processing
from Train import train


""" Directories paths and files names """
directory_raw_train_audio = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/train_audio"
train_birds_sample_dict_save = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/birds_samples/train_birds_2_sample_20_dict"
test_birds_sample_dict_save = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/birds_samples/test_birds_2_sample_20_dict"


""" Data management """
# Limits for the tests
birds_folder_limit = 2
samples_by_bird_limit = 10
data_split = 0.75


""" Hyperparameters """
learning_rate = 0.1
n_epochs = 100
batch_size = 8
max_audio_duration = 5000 # Max duration of the files in milliseconds


if __name__ == '__main__':
    train_sample_path = Path(train_birds_sample_dict_save + ".pkl")
    test_sample_path = Path(test_birds_sample_dict_save + ".pkl")

    # The process of loading and transforming data is long. Save the files after doing so
    if not (train_sample_path.exists()) or not(test_sample_path.exists()):
        print("=========================== Data preprocessing in progress...")
        # Get dictionary containing all the samples for each bird in the directory
        birds_samples_dict = data_pre_processing.get_birds_samples(directory_raw_train_audio, max_audio_duration, birds_folder_limit,
                                               samples_by_bird_limit)

        # Split the initial dictionary into train and test dictionary
        train_birds_samples_dict, test_birds_samples_dict = data_pre_processing.split_data_dict(birds_samples_dict, data_split)

        # Save the train and test dictionary
        data_pre_processing.save_audio_samples_from_dict(train_birds_samples_dict, train_birds_sample_dict_save)
        data_pre_processing.save_audio_samples_from_dict(test_birds_samples_dict, test_birds_sample_dict_save)
        print("=========================== Data preprocessing done.\n")

    print("=========================== Loading datasets...")
    # Use the pre-processed data in the datasets split in train and test
    train_dataset = dataset.BirdDataset(train_birds_sample_dict_save)
    test_dataset = dataset.BirdDataset(train_birds_sample_dict_save)
    print("=========================== Datasets loaded.\n")

    print("=========================== Training Started...")
    train.start_training(train_dataset, test_dataset, learning_rate, n_epochs, batch_size)
    print("=========================== End of training.\n")

