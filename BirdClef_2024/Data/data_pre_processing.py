import os
from pydub import AudioSegment
import numpy as np
import itertools
import pickle


def get_birds_samples(directory_path, max_audio_duration, birds_folder_limit=np.inf, samples_by_bird_limit=np.inf):
    # Dictionary to hold subfolders of the birds and the names of their audios
    birds_folders = get_birds_folders_and_files(directory_path, birds_folder_limit, samples_by_bird_limit)

    # Check that all the samples have the same sample rate
    check_samples_rates(directory_path, birds_folders)

    # Dictionary to hold bird's folders and the samples of their audio files
    birds_samples = get_birds_samples_from_dict_folders(directory_path, birds_folders, max_audio_duration)

    return birds_samples


def get_birds_folders_and_files(directory_path, birds_folder_limit, samples_by_bird_limit):
    # Dictionary to hold subfolders of the birds and the names of their audios
    birds_folders = {}

    # Iterate through each subfolder
    for f in itertools.islice(os.scandir(directory_path), birds_folder_limit):
        if f.is_dir():
            # Get the list of files in the subfolder
            files = [
                file.name
                for file in itertools.islice(os.scandir(f.path), samples_by_bird_limit)
                if file.is_file()
            ]

            # Add the subfolder and its files to the dictionary
            birds_folders[f.name] = files

    return birds_folders


def check_samples_rates(directory_path, birds_folders):
    birds_samples_rate = {}

    print("Checking audio files sample rate...")

    for one_bird_folder, files in birds_folders.items():
        for one_file in files:
            # Load the .ogg file
            path = directory_path + "/" + one_bird_folder + "/" + one_file
            audio = AudioSegment.from_ogg(path)

            # Get the sample rate (frame rate)
            sample_rate = audio.frame_rate

            if sample_rate not in birds_samples_rate:
                birds_samples_rate[sample_rate] = 0

            birds_samples_rate[sample_rate] += 1

    # If the audio files have different samples rate
    if len(birds_samples_rate.keys()) > 1:
        print("The files do not have all the same sample rate :")
        for n_audios, sample_rate in birds_samples_rate.items():
            print("Sample rate : ", sample_rate)
            print("Number of audio with this sample rate : ", n_audios)

        return
    else:
        print("All the audio files have the same sample rate : ", list(birds_samples_rate.keys())[0], " kHz")


def get_birds_samples_from_dict_folders(directory_path, birds_folders, audio_duration):
    """
    :param directory_path:
    :param birds_folders:
    :param audio_duration: in milliseconds
    :return: A dictionary, each key is a bird name, to each key is attached all the samples with a duration of EXACTLY
             audio_duration
    """
    birds_samples = {}

    for one_bird_folder, files in birds_folders.items():
        for one_file in files:
            # Load the .ogg file
            path = directory_path + "/" + one_bird_folder + "/" + one_file
            audio = AudioSegment.from_ogg(path)

            # Split the audio if it's longer than max_duration
            for i in range(0, len(audio), audio_duration):
                # Extract a 5-second chunk
                chunk = audio[i:i + audio_duration]

                # We want to sample to be exactly 5 seconds (5000 milliseconds)
                if len(chunk) == audio_duration:
                    # Convert the chunk to a raw array of samples
                    one_sample = np.array(chunk.get_array_of_samples())

                    if one_bird_folder not in birds_samples:
                        birds_samples[one_bird_folder] = []

                    birds_samples[one_bird_folder].append(one_sample)

    n_samples = sum(len(values) for values in birds_samples.values())
    print("Total number of audio files : ", n_samples)

    return birds_samples


def save_audio_samples_from_dict(birds_samples, file_path):
    # Save the dictionary to a pickle file
    with open(file_path + ".pkl", 'wb') as f:
        pickle.dump(birds_samples, f)

    print("File saved successfully at : " + file_path)


def load_audio_samples_from_pickle(file_path):
    # Load the dictionary from the pickle file
    with open(file_path + ".pkl", 'rb') as f:
        birds_samples = pickle.load(f)

    print("File loaded successfully : " + file_path)

    return birds_samples
