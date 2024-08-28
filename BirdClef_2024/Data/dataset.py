import numpy as np
from Data import data_pre_processing


class BirdDataset:

    def __init__(self, bird_sample_path):
        self.birds_samples = []
        self.birds_list = []

        birds_samples_dict = data_pre_processing.load_audio_samples_from_pickle(bird_sample_path)

        # Iterate over the dictionary
        for key, values in birds_samples_dict.items():
            # Extend the keys_list with the bird repeated for the length of its values
            # We want one bird name for each sample
            self.birds_list.extend([key] * len(values))
            # Extend the values_list with the values
            self.birds_samples.extend(values)

        self.birds_samples = transform(self.birds_samples)

        assert len(self.birds_list) == len(self.birds_samples)

    def get_item(self, index):
        print("TODO")
        #return  self.birds_samples[index], self.TBD


def transform(birds_samples):
    # Create a tensor holding all the samples
    new_birds_samples = np.stack(birds_samples)

    # Convert the stacked tensor from float32 to float64
    # If I stock the normalized values in a int32 array, it gets truncated to 0
    new_birds_samples = new_birds_samples.astype(np.float64)

    # Get max and max values of the samples to normalize the tensor values
    max_value = np.max(new_birds_samples)
    min_value = np.min(new_birds_samples)

    # Perform normalization
    # Using vectorized operations to normalize all values at once
    new_birds_samples[new_birds_samples > 0] /= (max_value - 1)
    new_birds_samples[new_birds_samples < 0] /= (-min_value + 1)  # Corrected to ensure scaling of negative values

    return new_birds_samples
