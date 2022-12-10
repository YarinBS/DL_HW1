import pickle
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


def plot_convergence_over_epochs(train_list: list, test_list: list, epochs: int, mode: str, model: int) -> None:
    plt.plot(range(1, epochs + 1), train_list)
    plt.plot(range(1, epochs + 1), test_list)
    plt.xlabel('Epochs')
    plt.ylabel(f'{mode}')
    plt.title(f"Model {model}'s {mode} over epochs")
    plt.legend(['Train', 'Test'])
    plt.show()


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_size, 512)
        self.blinear2 = BayesianLinear(512, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)


def main():
    # --- Model 1 - without randomization, trained on the full MNIST dataset ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 1 - without randomization, trained on the full MNIST dataset ---")
    model_1 = BayesianNeuralNetwork(input_size=28 * 28,
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
    plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=MINI_EPOCHS, mode='CE Loss', model=1)

    print(f'KL Divergence for model 1 is {kl_divergence_from_nn(model=model_1)}\n')

    # --- Model 2 - without randomization, trained on the first 200 MNIST examples ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print(" --- Model 2 - without randomization, trained on the first 200 MNIST examples ---")
    model_2 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_2.parameters())

    train_images, train_labels = next(iter(train_loader))
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
    plot_convergence_over_epochs(train_losses_2, test_losses_2, epochs=EPOCHS, mode='CE Loss', model=2)

    print(f'KL Divergence for model 2 is {kl_divergence_from_nn(model=model_2)}\n')

    # --- Model 3 - with Ber(0.5) labels, trained on the first 200 MNIST examples ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print(" --- Model 3 - with Ber(0.5) labels, trained on the first 200 MNIST examples ---")
    model_3 = BayesianNeuralNetwork(input_size=28 * 28,
                                    output_size=10)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_3.parameters())

    train_images, train_labels = next(iter(train_loader))  # Fetching the first 128 training samples
    train_images = train_images.view(-1, 28 * 28)
    train_labels = torch.randint(low=0, high=2, size=(200,))  # Using randint() for random label generation

    train_accuracies_3, train_losses_3, test_accuracies_3, test_losses_3 = [], [], [], []
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
    plot_convergence_over_epochs(train_losses_3, test_losses_3, epochs=EPOCHS, mode='CE Loss', model=3)

    print(f'KL Divergence for model 3 is {kl_divergence_from_nn(model=model_3)}\n')


if __name__ == '__main__':
    main()
