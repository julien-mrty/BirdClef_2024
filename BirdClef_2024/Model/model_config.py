import numpy as np
import Model.model_tools as tools


def ReLU(z):
    return np.maximum(z, 0)


def ReLU_prime(z):
    return np.where(z < 0, 0, 1)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip to avoid overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z) ** 2


# PDF output function for classification task
def softmax_regression(z):
    # Need to reshape the output to match the expected output shape
    z = tools.my_transpose(z)

    # Compute softmax
    exp_z = np.exp(z)
    softmax_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return softmax_output

# Layers with sigmoid activation functions
def he_init(num_neurons_previous_layer, num_neurons_current_layer):
    stddev = np.sqrt(2. / num_neurons_previous_layer)
    return np.random.randn(num_neurons_current_layer, num_neurons_previous_layer) * stddev

# Layers using tanh activation functions
def xavier_init(num_neurons_previous_layer, num_neurons_current_layer):
    stddev = np.sqrt(1. / num_neurons_previous_layer)
    return np.random.randn(num_neurons_current_layer, num_neurons_previous_layer) * stddev


def random_init(num_neurons_previous_layer, num_neurons_current_layer):
    return np.random.rand(num_neurons_current_layer, num_neurons_previous_layer)
