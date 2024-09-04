from Model import model
import numpy as np
import Model.model_config as model_config
import Train.train_tools as train_tools


def start_training(model, train_dataset, test_dataset, n_epochs, batch_size, directory_training_result_save, directory_model_save):

    print("=========================== Training Started...")

    """ Samples """
    n_training_data = len(train_dataset.get_input_samples())
    print("Number of training data : ", n_training_data)


    inital_prediction = model.forward_propagation(train_dataset.get_input_samples())
    print("inital_prediction : ", inital_prediction[0])
    print("Expected values : ", train_dataset.get_target_values()[0])
    print("inital_prediction : ", inital_prediction[20])
    print("Expected values : ", train_dataset.get_target_values()[20])
    print("inital_prediction : ", inital_prediction[40])
    print("Expected values : ", train_dataset.get_target_values()[40])

    # Train the NN
    model.train_and_test(
        n_epochs=n_epochs,
        batch_size=batch_size,
        train_input_feature=train_dataset.get_input_samples(),
        train_target_output=train_dataset.get_target_values(),
        test_input_feature=test_dataset.get_input_samples(),
        test_target_output=test_dataset.get_target_values()
    )

    # Print result after training (I test the two classes)
    final_pred = model.forward_propagation(train_dataset.get_input_samples())
    print("Final pred : ", final_pred[0])
    print("Final pred : ", final_pred[20])
    print("Final pred : ", final_pred[40])

    # Save the loss values of the model
    train_tools.save_training_logs(model, directory_training_result_save)

    # Plot the loss values recorded
    train_tools.plot_loss_records(directory_training_result_save, model.name)

    # Save the model
    train_tools.save_model(directory_model_save, model)
    print("=========================== End of training.\n")
