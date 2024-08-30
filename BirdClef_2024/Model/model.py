import numpy as np
import Model.model_tools as tools
import Model.model_config as config_func


#np.random.seed(42)  # For reproducibility


"""
Backprop OK after the adding of softmax ? No need for changes ? TBD...
"""


class ActivationFunctions:
    functions = {
        'relu': (config_func.ReLU, config_func.ReLU_prime),
        'sigmoid': (config_func.sigmoid, config_func.sigmoid_prime),
        'tanh': (config_func.tanh, config_func.tanh_prime)
    }

    @staticmethod
    def get(name):
        return ActivationFunctions.functions.get(name, (None, None))


class FullyConnectedLayer:
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function=None):
        """
        :param num_neurons_previous_layer:
        :param num_neurons_current_layer:
        :param output_function: The output function of the layer is an activation function if the layer is a hidden
                                layer or a softmax regression function if the layer is the last layer.
        """
        # He initialization for ReLU, Xavier initialization for others ???
        self.weights = init_function(num_neurons_previous_layer, num_neurons_current_layer)
        self.bias = np.zeros((num_neurons_current_layer, 1))
        self.z = None  # Output of the layer before the activation function
        self.a = None  # Output of the layer after the activation function
        self.output_function = output_function

    def layer_forward_propagation(self, neuron_input):
        self.z = np.dot(self.weights, neuron_input) + self.bias
        # Apply activation function only if it's provided
        self.a = self.output_function(self.z) if self.output_function else self.z
        return self.a


class OutputLayer(FullyConnectedLayer):
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function):
        super().__init__(num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function)


class LayerFactory:
    @staticmethod
    def create_layer(layer_type, num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function=None):
        if layer_type == 'output':
            return OutputLayer(num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function)
        else:
            return FullyConnectedLayer(num_neurons_previous_layer, num_neurons_current_layer, init_function, output_function)


class ClassificationFullyConnectedNeuralNetwork:
    def __init__(self, input_feature_size, n_neurons_by_layer, learning_rate, init_function, act_func="relu"):
        # Hyperparameters
        self.learning_rate = learning_rate

        # Layers
        self.input_feature_size = input_feature_size
        self.layers = []
        self.layers_output = []
        self.n_neurons_by_layer = n_neurons_by_layer
        self.num_layer = len(self.n_neurons_by_layer)
        self.initialization_function = init_function

        # Get activation functions
        self.activation_function, self.activation_function_prime = ActivationFunctions.get(act_func)

        # Initialization of NN layers
        self.initialize_layers()

    def initialize_layers(self):
        for i in range(self.num_layer):
            input_size = self.input_feature_size if i == 0 else self.n_neurons_by_layer[i - 1]
            layer_type = 'output' if i == self.num_layer - 1 else 'hidden'
            layer = LayerFactory.create_layer(layer_type, input_size, self.n_neurons_by_layer[i], self.initialization_function, self.activation_function if layer_type == 'hidden' else config_func.softmax_regression)
            self.layers.append(layer)

    def set_activation_function(self, act_func, act_func_prime):
        self.activation_function = act_func
        self.activation_function_prime = act_func_prime

        for layer in self.layers:
            layer.activation_function = self.activation_function

    def forward_propagation(self, input_feature):
        # Transpose the input feature to match the expected shapes in the layers
        previous_layer_output = input_feature.T
        for layer in self.layers:
            previous_layer_output = layer.layer_forward_propagation(previous_layer_output)

        # Return the output of the last layer
        return previous_layer_output

    def backpropagation(self, input_feature, target_output):
        n = input_feature.shape[0]  # Number of samples

        # Initialize delta
        deltas = [None] * self.num_layer

        # Last layer delta
        last_layer = self.layers[-1]
        deltas[-1] = last_layer.a - target_output

        # I transpose in the matrix in the softmax function to match the expected shape. I need to re-transpose here
        # to match the expected shape for backpropagation (error in delta matrix mul)
        deltas[-1] =  tools.my_transpose(deltas[-1])

        # Back-propagate through layers
        # First compute the deltas
        for i in range(self.num_layer - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            deltas[i] = np.dot(next_layer.weights.T, deltas[i + 1]) * self.activation_function_prime(layer.z)

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_activation = input_feature.T  # (input_size, m)
            else:
                input_activation = self.layers[i - 1].a  # (neurons_in_previous_layer, m)

            gradient_weight = np.dot(deltas[i], input_activation.T) / n
            gradient_bias = np.sum(deltas[i], axis=1, keepdims=True) / n

            # Update the weights and biases
            layer.weights -= self.learning_rate * gradient_weight
            layer.bias -= self.learning_rate * gradient_bias

    def train_one_epoch(self, batch_size, input_feature, target_output):
        n = input_feature.shape[0]
        if n != target_output.shape[0]:
            raise ValueError(f"Input and target do not match: {n} vs {target_output.shape[0]}.")

        for i in range(0, n, batch_size):
            # Slice the batches
            batch_input = input_feature[i:i + batch_size, :]
            batch_target = target_output[i:i + batch_size]

            # Perform forward propagation and backpropagation on the current batch
            self.forward_propagation(batch_input)
            self.backpropagation(batch_input, batch_target)

    def train(self, number_of_epochs, batch_size, input_feature, target_output):
        for _ in range(number_of_epochs):
            self.train_one_epoch(batch_size, input_feature, target_output)
