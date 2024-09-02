from Model import model
import numpy as np
import Model.model_config as model_config


def start_training(train_dataset, test_dataset, learning_rate, weight_decay, n_epochs, batch_size):

    """ Samples """
    n_training_data = len(train_dataset.get_input_samples())
    print("Number of training data : ", n_training_data)
    input_feature_size = len(train_dataset.get_item(0)[0])
    output_size = len(train_dataset.get_target_values()[0])

    """ Neural network """
    # Create the NN
    fullyConnectedNN = model.ClassificationFullyConnectedNeuralNetwork(
        input_feature_size=input_feature_size,
        n_neurons_by_layer=[4, output_size],
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        init_function=model_config.random_init,
        act_func="sigmoid")

    inital_prediction = fullyConnectedNN.forward_propagation(train_dataset.get_input_samples())
    print("inital_prediction : ", inital_prediction[0])
    print("Expected values : ", train_dataset.get_target_values()[0])
    print("inital_prediction : ", inital_prediction[20])
    print("Expected values : ", train_dataset.get_target_values()[20])
    print("inital_prediction : ", inital_prediction[40])
    print("Expected values : ", train_dataset.get_target_values()[40])

    # Train the NN
    fullyConnectedNN.train_and_test(
        n_epochs, batch_size, train_dataset.get_input_samples(), train_dataset.get_target_values(),
                           test_dataset.get_input_samples(), train_dataset.get_target_values())

    # Print result after training (I test the two classes)
    final_pred = fullyConnectedNN.forward_propagation(train_dataset.get_input_samples())
    print("Final pred : ", final_pred[0])
    print("Final pred : ", final_pred[20])
    print("Final pred : ", final_pred[40])
