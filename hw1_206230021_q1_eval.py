import winsound
import pickle
import torch

from hw1_206230021_q1_train import model_1_train_and_test, model_2_train_and_test, model_3_train_and_test, \
    model_4_train_and_test, model_5_train_and_test, fetch_MNIST_data

# --- Sound constants ---
DURATION = 1500  # milliseconds
FREQ = 750  # Hz


def evaluate():
    model_1_train_and_test()
    model_2_train_and_test()
    model_3_train_and_test()
    model_4_train_and_test()
    model_5_train_and_test()

    # -------
    # Or:

    # for i in range(5):
    #     model = pickle.load(open(f'./models/q2/q2_model_{i+1}.pkl', 'rb'))
    #     _, _, test_data, test_loader = fetch_MNIST_data()
    #
    #     accuracies = []
    #     for (test_images, test_labels) in test_loader:
    #         test_images = test_images.view(-1, 28 * 28)
    #         test_outputs = model(test_images)
    #         test_predictions = torch.argmax(test_outputs, dim=1)
    #         accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
    #
    #     print(f'Accuracy of model {i+1}: {100 * (sum(accuracies) / len(accuracies))}%')


def main():
    evaluate()

    # BEEP
    winsound.Beep(FREQ, DURATION)


if __name__ == '__main__':
    main()
