import pickle
import torch
from hw1_206230021_q2_train import fetch_MNIST_data


def evaluate():
    """
    Print the accuracy of each model.
    """
    for i in range(5):
        model = pickle.load(open(f'./models/q3/q3_model_{i + 1}.pkl', 'rb'))
        _, _, test_data, test_loader = fetch_MNIST_data()

        accuracies = []
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        print(f"Model {i + 1}'s accuracy: {round(100 * (sum(accuracies) / len(accuracies)), 3)}%")


def main():
    evaluate()


if __name__ == '__main__':
    main()
