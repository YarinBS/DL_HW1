import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.losses import kl_divergence_from_nn

from hw1_206230021_q1_train import plot_convergence_over_epochs

# --- Hyper-parameters ---
BATCH_SIZE = 200
MINI_EPOCHS = 10
EPOCHS = 100


def fetch_MNIST_data():
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform,
                                  download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


class LogisticRegressionClassifier(nn.Module):
    """
    Logistic regression is pretty much a sigmoid function activated on a linear transformation
    """

    def __init__(self, input_size, output_size):
        super(LogisticRegressionClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x_ = self.layer1(x)
        outputs = torch.sigmoid(x_)
        return outputs


def main():
    # --- Model 1 - Unregularized Logistic Regression ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 1 - Unregularized Logistic Regression ---")
    lr_model = LogisticRegressionClassifier(input_size=28 * 28,
                                            output_size=10)

    loss = nn.CrossEntropyLoss()
    unregularized_optimizer = torch.optim.Adam(lr_model.parameters(), weight_decay=0)

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            unregularized_optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

            train_outputs = lr_model(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Calculating and accumulating loss
            current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
            epoch_train_loss += current_train_loss

            # Weights update
            current_train_loss.backward()
            unregularized_optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = lr_model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_1.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_1.append(epoch_train_loss.item())
        test_accuracies_1.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_1.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, mode='Accuracy', model=1)
    plot_convergence_over_epochs(train_losses_1, test_losses_1, mode='CE Loss', model=1)

    # --- Model 2 - Regularized Logistic Regression ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 2 - Regularized Logistic Regression ---")
    regularized_lr_model = LogisticRegressionClassifier(input_size=28 * 28,
                                                        output_size=10)

    loss = nn.CrossEntropyLoss()
    regularized_optimizer = torch.optim.Adam(regularized_lr_model.parameters(), weight_decay=1)

    train_accuracies_2, train_losses_2, test_accuracies_2, test_losses_2 = [], [], [], []
    for i in range(EPOCHS):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            regularized_optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

            train_outputs = regularized_lr_model(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Calculating and accumulating loss
            current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
            epoch_train_loss += current_train_loss

            # Weights update
            current_train_loss.backward()
            regularized_optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = regularized_lr_model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_2.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_2.append(epoch_train_loss.item())
        test_accuracies_2.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_2.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_2, test_accuracies_2, mode='Accuracy', model=2)
    plot_convergence_over_epochs(train_losses_2, test_losses_2, mode='CE Loss', model=2)


if __name__ == '__main__':
    main()
