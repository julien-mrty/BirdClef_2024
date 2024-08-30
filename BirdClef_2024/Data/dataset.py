import numpy as np
from Data import data_pre_processing


class BirdDataset:

    def __init__(self, bird_sample_path):
        self.input_birds_samples = []
        self.target_birds_list = []
        self.target_values = []

        birds_samples_dict = data_pre_processing.load_audio_samples_from_pickle(bird_sample_path)

        # Iterate over the dictionary
        for key, values in birds_samples_dict.items():
            # Extend the keys_list with the bird repeated for the length of its values
            # We want one bird name for each sample
            self.target_birds_list.extend([key] * len(values))
            # Extend the values_list with the values
            self.input_birds_samples.extend(values)

        self.input_birds_samples, self.target_values = transform(self.input_birds_samples, self.target_birds_list)

        assert len(self.target_values) == len(self.input_birds_samples)

    def get_item(self, index):
        return  self.input_birds_samples[index], self.target_values[index]

    def get_input_samples(self):
        return self.input_birds_samples

    def get_target_values(self):
        return self.target_values


def transform(birds_samples, target_birds_list):
    new_birds_samples = prepare_input_samples(birds_samples)
    new_target_birds_list = prepare_target_values(target_birds_list)

    return new_birds_samples, new_target_birds_list


def prepare_input_samples(input_samples):
    # Create a tensor holding all the samples
    new_input_samples = np.stack(input_samples)

    # Convert the stacked tensor from float32 to float64
    # If I stock the normalized values in int32 array, it gets truncated to 0
    new_input_samples = new_input_samples.astype(np.float64)

    # Get max and max values of the samples to normalize the tensor values
    max_value = np.max(new_input_samples)
    min_value = np.min(new_input_samples)

    # Perform normalization
    # Using vectorized operations to normalize all values at once
    new_input_samples[new_input_samples > 0] /= (max_value - 1)
    new_input_samples[new_input_samples < 0] /= -(min_value + 1)  # Corrected to ensure scaling of negative values

    return new_input_samples


def prepare_target_values(target_values):
    # The output of my model is a list of length the number of birds in the dataset
    # All the birds are set at 0 except the right one

    # Get the unique birds and their corresponding indices
    _, indices = np.unique(target_values, return_inverse=True)

    # Get the number of unique birds and the number of samples
    n_birds = len(set(target_values))
    n_samples = len(target_values)

    # Create 2D array filled with zeros, each line represent a sample, each column represent a bird
    new_target_values = np.zeros((n_samples, n_birds))
    # Set the appropriate positions to 1
    new_target_values[np.arange(n_samples), indices] = 1

    return new_target_values
