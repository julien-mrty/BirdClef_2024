import Data
from Data import dataset
from pathlib import Path


""" Directories paths and files names """
directory_raw_train_audio = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/train_audio"
birds_samples_save = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/birds_samples/birds_2_samples_20"

# Limits for the tests
birds_folder_limit = 2
samples_by_bird_limit = 10

max_audio_duration = 5000


if __name__ == '__main__':
    birds_samples_path = Path(birds_samples_save + ".pkl")

    # The process of loading and transforming data is long. Save the files after doing so
    if not birds_samples_path.exists():
        birds_samples_dict = Data.data_pre_processing.get_birds_samples(directory_raw_train_audio, max_audio_duration, birds_folder_limit,
                                               samples_by_bird_limit)

        Data.data_pre_processing.save_audio_samples_from_dict(birds_samples_dict, birds_samples_save)

    # Use the pre-processed in the dataset
    train_dataset = dataset.BirdDataset(birds_samples_save)

    train_dataset.get_item(0)

