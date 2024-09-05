import numpy as np
from datetime import datetime
import Model.model as model
import math


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


def get_batch_accuracy(model_prediction, target_values):
    # Compute the predicted classes
    predicted_classes = np.argmax(model_prediction, axis=1)

    # Compute the true classes
    true_classes = np.argmax(target_values, axis=1)

    # Count the number of correct predictions
    good_pred = np.sum(predicted_classes == true_classes)

    # Calculate the number of wrong predictions (can also use len(model_prediction) - good_pred)
    wrong_pred = len(model_prediction) - good_pred

    return good_pred, wrong_pred


def print_batch_training_logs(loss, n_samples, n_epochs, current_epoch, batch_size, batch_number, batch_classification_report):
    n_batches = math.ceil(n_samples / batch_size)

    print(f"Train : Epoch [{current_epoch + 1}/{n_epochs}], Batch [{batch_number}/{n_batches}], "
          f"Batch loss: {loss[-1]:.4f}, ")
          #f"Batch Accuracy: {batch_classification_report['accuracy']:.4f}")


def print_validation_logs(avg_loss, classification_report):
    print(f"Validation : Average loss : {avg_loss:.4f}, "
          f"Accuracy: {classification_report['accuracy']:.4f}")


def print_epoch_training_logs(logger, n_epochs):
    avg_train_acc = np.mean(logger.get_values_from_reports("train_report", "accuracy"))
    avg_val_acc = np.mean(logger.get_values_from_reports("val_report", "accuracy"))

    avg_train_loss = np.mean(logger.get_history()['train_loss'])
    avg_val_loss = np.mean(logger.get_history()['val_loss'])

    print(f"============ Model average : Epoch [{logger.get_history()['epoch'][-1]}/{n_epochs}], "
          f"Model train loss: {avg_train_loss:.4f}, Model validation loss: {avg_val_loss:.4f}, "
          f"Model train accuracy: {avg_train_acc:.4f}, Model validation accuracy: {avg_val_acc:.4f}")


def get_predictions_and_target_indexes(model_hypothesis, target_values):
    return np.argmax(model_hypothesis, axis=1), np.argmax(target_values, axis=1)