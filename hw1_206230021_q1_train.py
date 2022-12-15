"""
blitz's Bayesian Neural Network documentation:
https://github.com/piEsposito/blitz-bayesian-deep-learning
"""

import pickle
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import winsound
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

# --- Hyper-parameters ---
BATCH_SIZE = 200
MINI_EPOCHS = 20
EPOCHS = 75
BIG_EPOCHS = 100

# --- Sound constants ---
DURATION = 1500  # milliseconds
FREQ = 750  # Hz


def fetch_MNIST_data(filter_labels=None, random_test_labels=False):
    """
    Fetching PyTorch's MNIST dataset.
    Can also filter the fetched training/test data by digits.
    :param filter_labels: None (default) or list of digits to keep in the training/test set
    :param random_test_labels: False (default). If True, sets Ber(0.5) labels (0 or 1) to the test data
    :return: train/test dataset/loader
    """
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not filter_labels:  # Get the entire dataset
        train_dataset = datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=True)

        test_dataset = datasets.MNIST(root='./data/',
                                      train=False,
                                      transform=transform,
                                      download=True)

        if random_test_labels:
            test_dataset.targets = torch.bernoulli(torch.full(test_dataset.targets.shape, 0.5))

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
        filter_indices = np.where(
            (train_dataset.targets == filter_labels[0]) | (train_dataset.targets == filter_labels[1]))
        train_dataset.data = train_dataset.data[filter_indices[0], :, :]
        train_dataset.targets = train_dataset.targets[filter_indices]
        # Changing the labels from 3 and 8 to 0 and 1 to make the problem binary
        train_dataset.targets = torch.tensor([1 if label == filter_labels[1] else 0 for label in train_dataset.targets])

        test_dataset = datasets.MNIST(root='./data/',
                                      train=False,
                                      transform=transform,
                                      download=True)

        # Filtering the test data
        filter_indices = np.where(
            (test_dataset.targets == filter_labels[0]) | (test_dataset.targets == filter_labels[1]))
        test_dataset.data = test_dataset.data[filter_indices[0], :, :]
        test_dataset.targets = test_dataset.targets[filter_indices]
        # Changing the labels from 3 and 8 to 0 and 1 to make the problem binary
        test_dataset.targets = torch.tensor([1 if label == filter_labels[1] else 0 for label in test_dataset.targets])

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


def plot_convergence_over_epochs(train_list: list, test_list: list, epochs: int, mode: str, model: int, question: int):
    plt.plot(range(1, epochs + 1), train_list)
    plt.plot(range(1, epochs + 1), test_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'{mode}')
    plt.title(f"Model {model}'s {mode} over epochs")
    plt.legend(['Train', 'Test'])
    if 'hw1_206230021_q1_train' in sys.modules:  # If this file is imported, don't save the plot
        plt.savefig(f"./plots/q{question}/q{question}_model_{model}_{mode.lower()}.png")
    plt.show()


def plot_model_kl_divergence(kl_values, model):
    plt.plot(range(1, len(kl_values) + 1), kl_values)
    plt.xlabel("Epochs")
    plt.ylabel("KL Value")
    plt.title(f"Model {model}'s KL Divergence over epochs")
    plt.savefig(f"./plots/q2/q2_model_{model}_kl_divergence.png")
    plt.show()


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_1_train_and_test(epochs=MINI_EPOCHS):
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 1 - without randomization, trained on the full MNIST dataset ---")
    model_1 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=150,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters())

    train_accuracies_1, test_accuracies_1 = [], []
    model1_kl_values = []
    for i in range(epochs):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

            loss = model_1.sample_elbo(inputs=train_images,
                                       labels=train_labels,
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=1 / 50000)

            train_outputs = model_1(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            # Calculating the accuracy of the current train batch
            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            optimizer.step()

        # # Evaluation after each epoch
        # for (test_images, test_labels) in test_loader:
        #     test_images = test_images.view(-1, 28 * 28)
        #     test_outputs = model_1(test_images)
        #     test_predictions = torch.argmax(test_outputs, dim=1)
        #     current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_1.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        # test_accuracies_1.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model1_kl_values.append(kl_divergence_from_nn(model=model_1).item())

    # # Plotting accuracy over epochs
    # plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=epochs, mode='Accuracy', model=1,
    #                              question=2)

    return model_1, model1_kl_values


def model_2_train_and_test(epochs=EPOCHS):
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print(" --- Model 2 - without randomization, trained on the first 200 examples ---")
    model_2 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=150,
                                    output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_2.parameters())

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)

    train_accuracies_2, test_accuracies_2 = [], []
    model2_kl_values = []
    for i in range(epochs):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        loss = model_2.sample_elbo(inputs=train_images,
                                   labels=train_labels,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=1 / 50000)

        train_outputs = model_2(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Weights update
        loss.backward()
        optimizer.step()

        # # Evaluation after each epoch
        # for (test_images, test_labels) in test_loader:
        #     test_images = test_images.view(-1, 28 * 28)
        #     test_outputs = model_2(test_images)
        #     test_predictions = torch.argmax(test_outputs, dim=1)
        #     current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_2.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        # test_accuracies_2.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model2_kl_values.append(kl_divergence_from_nn(model=model_2).item())

    # # Plotting accuracy over epochs
    # plot_convergence_over_epochs(train_accuracies_2, test_accuracies_2, epochs=epochs, mode='Accuracy', model=2,
    #                              question=2)

    return model_2, model2_kl_values


def model_3_train_and_test(epochs=MINI_EPOCHS):
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data(filter_labels=[3, 8])

    print("--- Model 3 - without randomization, trained on the 200 first 3's and 8's ---")
    model_3 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=150,
                                    output_size=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3.parameters())

    train_accuracies_3, test_accuracies_3 = [], []
    model3_kl_values = []

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)

    for i in range(epochs):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        loss = model_3.sample_elbo(inputs=train_images,
                                   labels=train_labels,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=1 / 50000)

        train_outputs = model_3(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Weights update
        loss.backward()
        optimizer.step()

        # # Evaluation after each epoch
        # for (test_images, test_labels) in test_loader:
        #     test_images = test_images.view(-1, 28 * 28)
        #     test_outputs = model_3(test_images)
        #     test_predictions = torch.argmax(test_outputs, dim=1)
        #     current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_3.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        # test_accuracies_3.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model3_kl_values.append(kl_divergence_from_nn(model=model_3).item())

    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_3, test_accuracies_3, epochs=epochs, mode='Accuracy', model=3,
    #                              question=2)

    return model_3, model3_kl_values


def model_4_train_and_test(epochs=MINI_EPOCHS):
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data(filter_labels=[3, 8])

    print("--- Model 4 - without randomization, trained on all 3's and 8's ---")
    model_4 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=150,
                                    output_size=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_4.parameters())

    train_accuracies_4, test_accuracies_4 = [], []
    model4_kl_values = []
    for i in range(epochs):  # Running EPOCH times over the entire dataset
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)  # Fitting the image

            optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            loss = model_4.sample_elbo(inputs=train_images,
                                       labels=train_labels,
                                       criterion=criterion,
                                       sample_nbr=3,
                                       complexity_cost_weight=1 / 50000)

            train_outputs = model_4(train_images)  # Getting model output for the current train batch
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            optimizer.step()

        # # Evaluation after each epoch
        # for (test_images, test_labels) in test_loader:
        #     test_images = test_images.view(-1, 28 * 28)
        #     test_outputs = model_4(test_images)
        #     test_predictions = torch.argmax(test_outputs, dim=1)
        #     current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_4.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        # test_accuracies_4.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model4_kl_values.append(kl_divergence_from_nn(model=model_4).item())

    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_4, test_accuracies_4, epochs=epochs, mode='Accuracy', model=4,
    #                              question=2)

    return model_4, model4_kl_values


def model_5_train_and_test(epochs=BIG_EPOCHS):
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data(random_test_labels=True)

    print(" --- Model 5 - Ber(0.5) labels, trained on the first 200 MNIST examples ---")
    model_5 = BayesianNeuralNetwork(input_size=28 * 28,
                                    hidden_size=150,
                                    output_size=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_5.parameters())

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 200 training samples
    train_images = train_images.view(-1, 28 * 28)
    train_labels = torch.bernoulli(torch.full((200,), 0.5)).long()

    train_accuracies_5, test_accuracies_5 = [], []
    model5_kl_values = []
    for i in range(epochs):
        print(f'Epoch {i + 1}...')
        epoch_train_loss, epoch_test_loss = 0, 0
        current_train_accuracies, current_test_accuracies = [], []

        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        loss = model_5.sample_elbo(inputs=train_images,
                                   labels=train_labels,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=1 / 50000)

        train_outputs = model_5(train_images)  # Getting model output for the current train batch
        train_predictions = torch.argmax(train_outputs, dim=1)

        current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

        # Weights update
        loss.backward()
        optimizer.step()

        # # Evaluation after each epoch
        # for (test_images, test_labels) in test_loader:
        #     test_images = test_images.view(-1, 28 * 28)
        #     test_outputs = model_5(test_images)
        #     test_predictions = torch.argmax(test_outputs, dim=1)
        #     current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_5.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        # test_accuracies_5.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model5_kl_values.append(kl_divergence_from_nn(model=model_5).item())

    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_5, test_accuracies_5, epochs=epochs, mode='Accuracy', model=5,
    #                              question=2)

    return model_5, model5_kl_values


def save_models(models: list):
    for i in range(len(models)):
        with open(f"./models/q2/q2_model_{i + 1}.pkl", "wb") as f:
            pickle.dump(models[i], f)


@variational_estimator
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

    models = []

    model_functions = [
        'model_1_train_and_test()',
        'model_2_train_and_test()',
        'model_3_train_and_test()',
        'model_4_train_and_test()',
        'model_5_train_and_test()'
    ]

    for i in range(len(model_functions)):
        net, values = eval(model_functions[i])  # The eval() function evaluates a given string as a function
        models.append(net)
        plot_model_kl_divergence(values, model=i + 1)
        model_kl_value = kl_divergence_from_nn(model=net)
        kl_div_values.append(model_kl_value)
        kl_div_per_param_values.append(model_kl_value / count_model_parameters(net))

    # Display KL Divergence values
    for i in range(5):
        with open('kl_values', 'a') as f:
            f.write(f'Model {i + 1} KL Divergence: {kl_div_values[i]}\n')
            f.write(f'Model {i + 1} KL Divergence per parameter: {kl_div_per_param_values[i]}\n')
        print(f"Model {i + 1}'s KL Divergence: {kl_div_values[i]}; Per parameter: {kl_div_per_param_values[i]}")

    # Save models to .pkl files
    save_models(models)

    winsound.Beep(FREQ, DURATION)


if __name__ == '__main__':
    main()
