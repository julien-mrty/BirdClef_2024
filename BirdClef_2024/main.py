from pathlib import Path
from Data import data_pre_processing
from Train import train
import Model.model_tools as model_tools
import Model.model_config as model_config
import Train.train_tools as train_tools


""" Directories paths and files names """
directory_raw_train_audio = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/train_audio"
train_birds_sample_dict_save = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/birds_samples/train_birds_2_sample_20_dict"
test_birds_sample_dict_save = "C:/Users/julie/Desktop/All/Important/Programmation/AI/BirdClef_Kaggle/Data/birds_samples/test_birds_2_sample_20_dict"
directory_training_result_save = "Training_results/Data/"
directory_model_save = "Training_results/Models/"
model_name = "Model01"


""" Data management """
# Limits for the tests
birds_folder_limit = 2
samples_by_bird_limit = 10
data_train_test_split = 0.75


""" Hyperparameters """
learning_rate = 1e-1
weight_decay = 1e-6
n_epochs = 10
batch_size = 80
max_audio_duration = 5000 # Max duration of the files in milliseconds
init_function = model_config.random_init
act_func = "sigmoid"


if __name__ == '__main__':
    """ Data pre-processing """
    data_pre_processing.start_data_preprocessing(train_birds_sample_dict_save, test_birds_sample_dict_save,
                                                 directory_raw_train_audio, max_audio_duration, birds_folder_limit,
                                                 samples_by_bird_limit, data_train_test_split)

    """ Datasets loading """
    train_dataset, test_dataset = train_tools.load_datasets(train_birds_sample_dict_save, test_birds_sample_dict_save)

    """ Neural network initialization """
    model = model_tools.create_classification_fully_connected_nn(model_name, train_dataset, learning_rate, weight_decay,
                                                                 init_function, act_func)

    """ Training """
    train.start_training(model, train_dataset, test_dataset, n_epochs, batch_size, directory_training_result_save, directory_model_save)


