import numpy as np
from datetime import datetime
import Model.model as model


def my_transpose(output):
    if output.ndim == 1:
        # If it's a 1D vector, reshape it to (1, dim_output) to make it 2D and then transpose
        output = output.reshape(1, -1).T
    elif output.ndim == 2:
        # If it's already 2D, simply transpose it
        output = output.T
    else:
        raise ValueError("Output should be a 1D or 2D array.")

    return output


def compute_least_squared_cost_function(model_hypothesis, target_values):
    # Compute the difference, square it, sum over all features, and take the mean
    diff = model_hypothesis - target_values
    loss = 1 / 2 * np.mean(np.sum(diff ** 2, axis=1))

    return loss


def generate_model_name(name, learning_rate, weight_decay):
    # Get the current date and time
    current_datetime = datetime.now()

    # Extract just the date
    current_date = current_datetime.date()

    full_name = str(current_date) + "__name=" + name + "__lr=" + str(learning_rate) + "__wd=" + str(weight_decay)

    return full_name


def create_classification_fully_connected_nn(model_name, train_dataset, learning_rate, weight_decay, init_function, act_func):
    input_feature_size = len(train_dataset.get_item(0)[0])
    output_size = len(train_dataset.get_target_values()[0])

    my_model = model.ClassificationFullyConnectedNeuralNetwork(
        model_name=model_name,
        input_feature_size=input_feature_size,
        n_neurons_by_layer=[4, output_size],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        init_function=init_function,
        act_func=act_func)

    print("=========================== Neural network created.\n")

    return my_model