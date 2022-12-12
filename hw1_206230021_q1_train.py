"""
blitz's Bayesian Neural Network documentation:
https://github.com/piEsposito/blitz-bayesian-deep-learning
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from blitz.losses import kl_divergence_from_nn

# --- Hyper-parameters ---
BATCH_SIZE = 200
MINI_EPOCHS = 10
EPOCHS = 75


def fetch_MNIST_data(filter=None):
    """
    Fetching PyTorch's MNIST dataset.
    Can also filter the fetched training/test data by digits.
    :param filter: None (default) or list of digits to keep in the training/test set
    :return: train/test dataset/loader
    """
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not filter:  # Get the entire dataset
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

    else:  # Filter the training set
        train_dataset = datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=True)

        # Filtering the training data
        filter_indices = np.where((train_dataset.targets == filter[0]) | (train_dataset.targets == filter[1]))
        # filter_indices = np.where(train_dataset.targets in filter)
        train_dataset.data = train_dataset.data[filter_indices[0], :, :]
        train_dataset.targets = train_dataset.targets[filter_indices]

        test_dataset = datasets.MNIST(root='./data/',
                                      train=False,
                                      transform=transform,
                                      download=True)

        # Filtering the test data
        filter_indices = np.where((test_dataset.targets == filter[0]) | (test_dataset.targets == filter[1]))
        # filter_indices = np.where(train_dataset.targets in filter)
        test_dataset.data = test_dataset.data[filter_indices[0], :, :]
        test_dataset.targets = test_dataset.targets[filter_indices]

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


def plot_convergence_over_epochs(train_list: list, test_list: list, epochs: int, mode: str, model: int) -> None:
    plt.plot(range(1, epochs + 1), train_list)
    plt.plot(range(1, epochs + 1), test_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'{mode}')
    plt.title(f"Model {model}'s {mode} over epochs")
    plt.legend(['Train', 'Test'])
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_1_train_and_eval():
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 1 - without randomization, trained on the full MNIST dataset ---")
    model_1 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=512,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters())

    train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

            train_outputs = model_1(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Calculating and accumulating loss
            current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
            epoch_train_loss += current_train_loss

            # Weights update
            current_train_loss.backward()
            optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_1(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_1.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_1.append(epoch_train_loss.item())
        test_accuracies_1.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_1.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=MINI_EPOCHS, mode='Accuracy', model=1)
    # plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=MINI_EPOCHS, mode='CE Loss', model=1)

    return model_1


def model_2_train_and_eval():
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print(" --- Model 2 - without randomization, trained on the first 200 examples ---")
    model_2 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=512,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_2.parameters())

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)

    train_accuracies_2, train_losses_2, test_accuracies_2, test_losses_2 = [], [], [], []
    for i in range(EPOCHS):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        train_outputs = model_2(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Calculating and accumulating loss
        current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
        epoch_train_loss += current_train_loss

        # Weights update
        current_train_loss.backward()
        optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_2(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_2.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_2.append(epoch_train_loss.item())
        test_accuracies_2.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_2.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_2, test_accuracies_2, epochs=EPOCHS, mode='Accuracy', model=2)
    # plot_convergence_over_epochs(train_losses_2, test_losses_2, epochs=EPOCHS, mode='CE Loss', model=2)

    return model_2


def model_3_train_and_eval():
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data([3, 8])

    print("--- Model 3 - without randomization, trained on the 200 first 3's and 8's ---")
    model_3 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=512,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3.parameters())

    train_accuracies_3, train_losses_3, test_accuracies_3, test_losses_3 = [], [], [], []

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)

    for i in range(EPOCHS):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        train_outputs = model_3(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Calculating and accumulating loss
        current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
        epoch_train_loss += current_train_loss

        # Weights update
        current_train_loss.backward()
        optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_3(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_3.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_3.append(epoch_train_loss.item())
        test_accuracies_3.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_3.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_3, test_accuracies_3, epochs=EPOCHS, mode='Accuracy', model=3)
    # plot_convergence_over_epochs(train_losses_3, test_losses_3, epochs=EPOCHS, mode='CE Loss', model=3)

    return model_3


def model_4_train_and_eval():
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data([3, 8])

    print("--- Model 4 - without randomization, trained on all 3's and 8's ---")
    model_4 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=512,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_4.parameters())

    train_accuracies_4, train_losses_4, test_accuracies_4, test_losses_4 = [], [], [], []
    for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

            train_outputs = model_4(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Calculating and accumulating loss
            current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
            epoch_train_loss += current_train_loss

            # Weights update
            current_train_loss.backward()
            optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = model_4(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_4.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_4.append(epoch_train_loss.item())
        test_accuracies_4.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_4.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_4, test_accuracies_4, epochs=MINI_EPOCHS, mode='Accuracy', model=4)
    # plot_convergence_over_epochs(train_losses_4, test_losses_4, epochs=MINI_EPOCHS, mode='CE Loss', model=4)

    return model_4


def model_5_train_and_eval():
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print(" --- Model 5 - Ber(0.5) labels, trained on the first 200 MNIST examples ---")
    model_5 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=512,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_5.parameters())

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)
    train_labels = torch.bernoulli(torch.full((200,), 0.5))

    train_accuracies_5, train_losses_5, test_accuracies_5, test_losses_5 = [], [], [], []
    for i in range(EPOCHS):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

        train_outputs = model_5(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Calculating and accumulating loss
        current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
        epoch_train_loss += current_train_loss

        # Weights update
        current_train_loss.backward()
        optimizer.step()

        # Evaluation after each epoch
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_labels = torch.bernoulli(torch.full((200,), 0.5))
            test_outputs = model_5(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
            current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
            epoch_test_loss += current_test_loss

        train_accuracies_5.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        train_losses_5.append(epoch_train_loss.item())
        test_accuracies_5.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
        test_losses_5.append(epoch_test_loss.item())

    # Plotting accuracy and loss graphs
    plot_convergence_over_epochs(train_accuracies_5, test_accuracies_5, epochs=EPOCHS, mode='Accuracy', model=5)
    # plot_convergence_over_epochs(train_losses_5, test_losses_5, epochs=EPOCHS, mode='CE Loss', model=5)

    return model_5


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.blinear1 = BayesianLinear(input_size, hidden_size)
        self.blinear2 = BayesianLinear(hidden_size, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)


def main():
    kl_div_values = []
    kl_div_per_param_values = []

    model1 = model_1_train_and_eval()  # --- Model 1 - without randomization, trained on the full MNIST dataset ---
    kl_div_value = kl_divergence_from_nn(model=model1)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model1))

    model2 = model_2_train_and_eval()  # --- Model 2 - without randomization, trained on the first 200 examples ---
    kl_div_value = kl_divergence_from_nn(model=model2)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model2))

    model3 = model_3_train_and_eval()  # --- Model 3 - without randomization, trained on the 200 first 3's and 8's ---
    kl_div_value = kl_divergence_from_nn(model=model3)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model3))

    model4 = model_4_train_and_eval()  # -- Model 4 - without randomization, trained on all 3's and 8's ---
    kl_div_value = kl_divergence_from_nn(model=model4)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model4))

    model5 = model_5_train_and_eval()  # --- Model 5 - Ber(0.5) labels, trained on the first 200 MNIST examples ---
    kl_div_value = kl_divergence_from_nn(model=model5)
    kl_div_values.append(kl_div_value)
    kl_div_per_param_values.append(kl_div_value / count_parameters(model5))

    # Display KL Divergence values
    for i in range(5):
        print(f"Model {i + 1}'s KL Divergence: {kl_div_values[i]}; Per parameter: {kl_div_per_param_values[i]}")

    # TODO: Plot a graph of KL values over epochs


if __name__ == '__main__':
    main()
