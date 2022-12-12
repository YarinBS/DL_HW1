import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear

from hw1_206230021_q1_train import fetch_MNIST_data, plot_convergence_over_epochs, BayesianNeuralNetwork

# --- Hyper-parameters ---
BATCH_SIZE = 200
MINI_EPOCHS = 15
EPOCHS = 75


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
        # return x_


class DeepBayesianNeuralNetwork(nn.Module):
    """
    Just like the BayesianNeuralNetwork, but with an extra Bayesian layers
    """

    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_size, hidden1)
        self.blinear2 = BayesianLinear(hidden1, hidden2)
        self.blinear3 = BayesianLinear(hidden2, output_size)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        x_ = self.blinear2(x_)
        x_ = F.relu(x_)
        return self.blinear3(x_)


def main():
    # # --- Model 1 - Unregularized Logistic Regression ---
    # train_data, train_loader, test_data, test_loader = fetch_MNIST_data()
    #
    # print("--- Model 1 - Unregularized Logistic Regression ---")
    # lr_model = LogisticRegressionClassifier(input_size=28 * 28,
    #                                         output_size=10)
    #
    # loss = nn.CrossEntropyLoss()
    # unregularized_optimizer = torch.optim.SGD(lr_model.parameters(), weight_decay=0, lr=0.001)
    #
    # train_accuracies_1, train_losses_1, test_accuracies_1, test_losses_1 = [], [], [], []
    # for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
    #     print(f'Epoch {i + 1}...')
    #     epoch_train_loss, epoch_test_loss = 0, 0
    #     current_train_accuracies, current_test_accuracies = [], []
    #     # Training phase
    #     for (train_images, train_labels) in train_loader:
    #         # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
    #         train_images = train_images.view(-1, 28 * 28)  # Fitting the image
    #
    #         unregularized_optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    #
    #         train_outputs = lr_model(train_images)  # Getting model output for the current train batch
    #         train_predictions = torch.argmax(train_outputs, dim=1)
    #
    #         current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))
    #
    #         # Calculating and accumulating loss
    #         current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
    #         epoch_train_loss += current_train_loss
    #
    #         # Weights update
    #         current_train_loss.backward()
    #         unregularized_optimizer.step()
    #
    #     # Evaluation after each epoch
    #     for (test_images, test_labels) in test_loader:
    #         test_images = test_images.view(-1, 28 * 28)
    #         test_outputs = lr_model(test_images)
    #         test_predictions = torch.argmax(test_outputs, dim=1)
    #         current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
    #         current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
    #         epoch_test_loss += current_test_loss
    #
    #     train_accuracies_1.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
    #     train_losses_1.append(epoch_train_loss.item())
    #     test_accuracies_1.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
    #     test_losses_1.append(epoch_test_loss.item())
    #
    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_1, test_accuracies_1, epochs=MINI_EPOCHS, mode='Accuracy', model=1)
    # # plot_convergence_over_epochs(train_losses_1, test_losses_1, epochs=MINI_EPOCHS, mode='CE Loss', model=1)

    # --- Model 2 - Regularized Logistic Regression ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 2 - Regularized Logistic Regression ---")
    regularized_lr_model = LogisticRegressionClassifier(input_size=28 * 28,
                                                        output_size=10)

    loss = nn.CrossEntropyLoss()
    regularized_optimizer = torch.optim.SGD(regularized_lr_model.parameters(), weight_decay=1, lr=0.001)

    train_accuracies_2, train_losses_2, test_accuracies_2, test_losses_2 = [], [], [], []
    for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
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

    plot_convergence_over_epochs(train_accuracies_2, test_accuracies_2, epochs=MINI_EPOCHS, mode='Accuracy', model=2)
    # plot_convergence_over_epochs(train_losses_2, test_losses_2, epochs=MINI_EPOCHS, mode='CE Loss', model=2)

    # # --- Model 3 - BNN ---
    # train_data, train_loader, test_data, test_loader = fetch_MNIST_data()
    #
    # print("--- Model 3 - BNN 28*28 -> 512 -> 10 ---")
    # bnn = BayesianNeuralNetwork(input_size=28*28,
    #                             hidden_size=512,
    #                             output_size=10)
    #
    # loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(bnn.parameters())
    #
    # train_accuracies_3, train_losses_3, test_accuracies_3, test_losses_3 = [], [], [], []
    # for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
    #     print(f'Epoch {i + 1}...')
    #     epoch_train_loss, epoch_test_loss = 0, 0
    #     current_train_accuracies, current_test_accuracies = [], []
    #     # Training phase
    #     for (train_images, train_labels) in train_loader:
    #         # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
    #         train_images = train_images.view(-1, 28 * 28)  # Fitting the image
    #
    #         optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    #
    #         train_outputs = bnn(train_images)  # Getting model output for the current train batch
    #         train_predictions = torch.argmax(train_outputs, dim=1)
    #
    #         current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))
    #
    #         # Calculating and accumulating loss
    #         current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
    #         epoch_train_loss += current_train_loss
    #
    #         # Weights update
    #         current_train_loss.backward()
    #         optimizer.step()
    #
    #     # Evaluation after each epoch
    #     for (test_images, test_labels) in test_loader:
    #         test_images = test_images.view(-1, 28 * 28)
    #         test_outputs = bnn(test_images)
    #         test_predictions = torch.argmax(test_outputs, dim=1)
    #         current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
    #         current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
    #         epoch_test_loss += current_test_loss
    #
    #     train_accuracies_3.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
    #     train_losses_3.append(epoch_train_loss.item())
    #     test_accuracies_3.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
    #     test_losses_3.append(epoch_test_loss.item())
    #
    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_3, test_accuracies_3, epochs=MINI_EPOCHS, mode='Accuracy', model=3)
    # # plot_convergence_over_epochs(train_losses_3, test_losses_3, epochs=MINI_EPOCHS, mode='CE Loss', model=3)
    #
    # # --- Model 4 - Deep BNN with hidden1 = 512, hidden2 = 100 ---
    # train_data, train_loader, test_data, test_loader = fetch_MNIST_data()
    #
    # print("--- Model 4 - DBNN 28*28 -> 512 -> 100 -> 10 ---")
    # dbnn1 = DeepBayesianNeuralNetwork(input_size=28 * 28,
    #                                   hidden1=512,
    #                                   hidden2=100,
    #                                   output_size=10)
    #
    # loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(dbnn1.parameters())
    #
    # train_accuracies_4, train_losses_4, test_accuracies_4, test_losses_4 = [], [], [], []
    # for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
    #     print(f'Epoch {i + 1}...')
    #     epoch_train_loss, epoch_test_loss = 0, 0
    #     current_train_accuracies, current_test_accuracies = [], []
    #     # Training phase
    #     for (train_images, train_labels) in train_loader:
    #         # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
    #         train_images = train_images.view(-1, 28 * 28)  # Fitting the image
    #
    #         optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    #
    #         train_outputs = dbnn1(train_images)  # Getting model output for the current train batch
    #         train_predictions = torch.argmax(train_outputs, dim=1)
    #
    #         current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))
    #
    #         # Calculating and accumulating loss
    #         current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
    #         epoch_train_loss += current_train_loss
    #
    #         # Weights update
    #         current_train_loss.backward()
    #         optimizer.step()
    #
    #     # Evaluation after each epoch
    #     for (test_images, test_labels) in test_loader:
    #         test_images = test_images.view(-1, 28 * 28)
    #         test_outputs = dbnn1(test_images)
    #         test_predictions = torch.argmax(test_outputs, dim=1)
    #         current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
    #         current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
    #         epoch_test_loss += current_test_loss
    #
    #     train_accuracies_4.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
    #     train_losses_4.append(epoch_train_loss.item())
    #     test_accuracies_4.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
    #     test_losses_4.append(epoch_test_loss.item())
    #
    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_4, test_accuracies_4, epochs=MINI_EPOCHS, mode='Accuracy', model=4)
    # # plot_convergence_over_epochs(train_losses_4, test_losses_4, epochs=MINI_EPOCHS, mode='CE Loss', model=4)
    #
    # # --- Model 5 - Deep BNN with hidden1 = 250, hidden2 = 50 ---
    # train_data, train_loader, test_data, test_loader = fetch_MNIST_data()
    #
    # print("--- Model 5 - DBNN 28*28 -> 250 -> 50 -> 10 ---")
    # dbnn2 = DeepBayesianNeuralNetwork(input_size=28 * 28,
    #                                   hidden1=250,
    #                                   hidden2=50,
    #                                   output_size=10)
    #
    # loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(dbnn2.parameters())
    #
    # train_accuracies_5, train_losses_5, test_accuracies_5, test_losses_5 = [], [], [], []
    # for i in range(MINI_EPOCHS):  # Running EPOCH times over the entire dataset
    #     print(f'Epoch {i + 1}...')
    #     epoch_train_loss, epoch_test_loss = 0, 0
    #     current_train_accuracies, current_test_accuracies = [], []
    #     # Training phase
    #     for (train_images, train_labels) in train_loader:
    #         # train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
    #         train_images = train_images.view(-1, 28 * 28)  # Fitting the image
    #
    #         optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    #
    #         train_outputs = dbnn2(train_images)  # Getting model output for the current train batch
    #         train_predictions = torch.argmax(train_outputs, dim=1)
    #
    #         current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))
    #
    #         # Calculating and accumulating loss
    #         current_train_loss = loss(input=train_outputs.float(), target=train_labels.long())
    #         epoch_train_loss += current_train_loss
    #
    #         # Weights update
    #         current_train_loss.backward()
    #         optimizer.step()
    #
    #     # Evaluation after each epoch
    #     for (test_images, test_labels) in test_loader:
    #         test_images = test_images.view(-1, 28 * 28)
    #         test_outputs = dbnn2(test_images)
    #         test_predictions = torch.argmax(test_outputs, dim=1)
    #         current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))
    #         current_test_loss = loss(input=test_outputs.float(), target=test_labels.long())
    #         epoch_test_loss += current_test_loss
    #
    #     train_accuracies_5.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
    #     train_losses_5.append(epoch_train_loss.item())
    #     test_accuracies_5.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))
    #     test_losses_5.append(epoch_test_loss.item())
    #
    # # Plotting accuracy and loss graphs
    # plot_convergence_over_epochs(train_accuracies_5, test_accuracies_5, epochs=MINI_EPOCHS, mode='Accuracy', model=5)
    # # plot_convergence_over_epochs(train_losses_5, test_losses_5, epochs=MINI_EPOCHS, mode='CE Loss', model=5)

    import winsound
    duration = 2000  # milliseconds
    freq = 600  # Hz
    winsound.Beep(freq, duration)


if __name__ == '__main__':
    main()
