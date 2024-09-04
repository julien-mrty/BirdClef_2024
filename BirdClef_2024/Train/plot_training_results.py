import Train.train_tools as train_tools


directory_training_result_save = "Training_results/Data/"
model_name = "2024-09-04__name=Model01__lr=0.1__wd=1e-06"


def main():
    # Plot the loss values recorded
    train_tools.plot_loss_records(directory_training_result_save, model_name)

if __name__ == '__main__':
    main()