import torch
import torch.nn as nn
import torch.nn.functional as F
import winsound
from blitz.losses import kl_divergence_from_nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from hw1_206230021_q1_train import fetch_MNIST_data, plot_convergence, BayesianNeuralNetwork, save_models

# --- Hyper-parameters ---
EPOCHS = 15

# --- Sound constants ---
DURATION = 1500  # milliseconds
FREQ = 750  # Hz


def model_1_train_and_test(epochs=EPOCHS):
    # --- Model 1 - Unregularized Logistic Regression ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 1 - Unregularized Logistic Regression ---")
    lr_model = LogisticRegressionClassifier(input_size=28 * 28,
                                            output_size=10)

    criterion = nn.CrossEntropyLoss()
    unregularized_optimizer = torch.optim.SGD(lr_model.parameters(), weight_decay=0, lr=0.001)

    train_accuracies_1, test_accuracies_1 = [], []
    model1_kl_values = []
    for i in range(epochs):
        print(f'Epoch [{i + 1}/{epochs}]...', end=' ')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)

            unregularized_optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            loss = lr_model.sample_elbo(inputs=train_images,
                                        labels=train_labels,
                                        criterion=criterion,
                                        sample_nbr=3,
                                        complexity_cost_weight=1 / 50000)

            train_outputs = lr_model(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            unregularized_optimizer.step()

        # Evaluation
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = lr_model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_1.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        test_accuracies_1.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model1_kl_values.append(kl_divergence_from_nn(model=lr_model))

        print(f"Train Accuracy: {round(train_accuracies_1[-1], 3)}%, Test Accuracy: {round(test_accuracies_1[-1], 3)}%")

    # Plotting accuracy and KL divergence over epochs
    plot_convergence(train_or_kl_list=train_accuracies_1, test_list=test_accuracies_1, epochs=epochs, mode='Accuracy',
                     model=1, question=3)
    plot_convergence(train_or_kl_list=model1_kl_values, test_list=[], epochs=epochs, mode='KL Divergence', model=1,
                     question=3)

    return lr_model


def model_2_train_and_test(epochs=EPOCHS):
    # --- Model 2 - Regularized Logistic Regression ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 2 - Regularized Logistic Regression ---")
    regularized_lr_model = LogisticRegressionClassifier(input_size=28 * 28,
                                                        output_size=10)

    criterion = nn.CrossEntropyLoss()
    regularized_optimizer = torch.optim.SGD(regularized_lr_model.parameters(), weight_decay=1, lr=0.001)

    train_accuracies_2, test_accuracies_2 = [], []
    model2_kl_values = []
    for i in range(epochs):
        print(f'Epoch [{i + 1}/{epochs}]...', end=' ')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)

            regularized_optimizer.zero_grad()
            loss = regularized_lr_model.sample_elbo(inputs=train_images,
                                                    labels=train_labels,
                                                    criterion=criterion,
                                                    sample_nbr=3,
                                                    complexity_cost_weight=1 / 50000)

            train_outputs = regularized_lr_model(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            regularized_optimizer.step()

        # Evaluation
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = regularized_lr_model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_2.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        test_accuracies_2.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model2_kl_values.append(kl_divergence_from_nn(model=regularized_lr_model))

        print(f"Train Accuracy: {round(train_accuracies_2[-1], 3)}%, Test Accuracy: {round(test_accuracies_2[-1], 3)}%")

    # Plotting accuracy and KL divergence over epochs
    plot_convergence(train_or_kl_list=train_accuracies_2, test_list=test_accuracies_2, epochs=epochs, mode='Accuracy',
                     model=2, question=3)
    plot_convergence(train_or_kl_list=model2_kl_values, test_list=[], epochs=epochs, mode='KL Divergence', model=2,
                     question=3)

    return regularized_lr_model


def model_3_train_and_test(epochs=EPOCHS):
    # --- Model 3 - BNN ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 3 - BNN 28*28 -> 150 -> 10 ---")
    bnn = BayesianNeuralNetwork(input_size=28 * 28,
                                hidden_size=150,
                                output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bnn.parameters())

    train_accuracies_3, test_accuracies_3 = [], []
    model3_kl_values = []
    for i in range(epochs):
        print(f'Epoch [{i + 1}/{epochs}]...', end=' ')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)

            optimizer.zero_grad()
            loss = bnn.sample_elbo(inputs=train_images,
                                   labels=train_labels,
                                   criterion=criterion,
                                   sample_nbr=3,
                                   complexity_cost_weight=1 / 50000)

            train_outputs = bnn(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            optimizer.step()

        # Evaluation
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = bnn(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_3.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        test_accuracies_3.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model3_kl_values.append(kl_divergence_from_nn(model=bnn).item())

        print(f"Train Accuracy: {round(train_accuracies_3[-1], 3)}%, Test Accuracy: {round(test_accuracies_3[-1], 3)}%")

    # Plotting accuracy and KL divergence over epochs
    plot_convergence(train_or_kl_list=train_accuracies_3, test_list=test_accuracies_3, epochs=epochs, mode='Accuracy',
                     model=3, question=3)
    plot_convergence(train_or_kl_list=model3_kl_values, test_list=[], epochs=epochs, mode='KL Divergence', model=3,
                     question=3)

    return bnn


def model_4_train_and_test(epochs=EPOCHS):
    # --- Model 4 - Deep BNN with hidden1 = 150, hidden2 = 100 ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 4 - DBNN 28*28 -> 150 -> 100 -> 10 ---")
    dbnn1 = DeepBayesianNeuralNetwork(input_size=28 * 28,
                                      hidden1=150,
                                      hidden2=100,
                                      output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dbnn1.parameters())

    train_accuracies_4, test_accuracies_4 = [], []
    model4_kl_values = []
    for i in range(epochs):
        print(f'Epoch [{i + 1}/{epochs}]...', end=' ')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)

            optimizer.zero_grad()
            loss = dbnn1.sample_elbo(inputs=train_images,
                                     labels=train_labels,
                                     criterion=criterion,
                                     sample_nbr=3,
                                     complexity_cost_weight=1 / 50000)

            train_outputs = dbnn1(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            optimizer.step()

        # Evaluation
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = dbnn1(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_4.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        test_accuracies_4.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model4_kl_values.append(kl_divergence_from_nn(model=dbnn1).item())

        print(f"Train Accuracy: {round(train_accuracies_4[-1], 3)}%, Test Accuracy: {round(test_accuracies_4[-1], 3)}%")

    # Plotting accuracy and KL divergence over epochs
    plot_convergence(train_or_kl_list=train_accuracies_4, test_list=test_accuracies_4, epochs=epochs, mode='Accuracy',
                     model=4, question=3)
    plot_convergence(train_or_kl_list=model4_kl_values, test_list=[], epochs=epochs, mode='KL Divergence', model=4,
                     question=3)

    return dbnn1


def model_5_train_and_test(epochs=EPOCHS):
    # --- Model 5 - Deep BNN with hidden1 = 150, hidden2 = 50 ---
    train_data, train_loader, test_data, test_loader = fetch_MNIST_data()

    print("--- Model 5 - DBNN 28*28 -> 150 -> 50 -> 10 ---")
    dbnn2 = DeepBayesianNeuralNetwork(input_size=28 * 28,
                                      hidden1=150,
                                      hidden2=50,
                                      output_size=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dbnn2.parameters())

    train_accuracies_5, test_accuracies_5 = [], []
    model5_kl_values = []
    for i in range(epochs):
        print(f'Epoch [{i + 1}/{epochs}]...', end=' ')
        current_train_accuracies, current_test_accuracies = [], []
        # Training phase
        for (train_images, train_labels) in train_loader:
            train_images = train_images.view(-1, 28 * 28)

            optimizer.zero_grad()
            loss = dbnn2.sample_elbo(inputs=train_images,
                                     labels=train_labels,
                                     criterion=criterion,
                                     sample_nbr=3,
                                     complexity_cost_weight=1 / 50000)

            train_outputs = dbnn2(train_images)
            train_predictions = torch.argmax(train_outputs, dim=1)

            current_train_accuracies.append(((train_predictions == train_labels).sum().item()) / train_labels.size(0))

            # Weights update
            loss.backward()
            optimizer.step()

        # Evaluation
        for (test_images, test_labels) in test_loader:
            test_images = test_images.view(-1, 28 * 28)
            test_outputs = dbnn2(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            current_test_accuracies.append(((test_predictions == test_labels).sum().item()) / test_labels.size(0))

        train_accuracies_5.append(100 * (sum(current_train_accuracies) / len(current_train_accuracies)))
        test_accuracies_5.append(100 * (sum(current_test_accuracies) / len(current_test_accuracies)))

        model5_kl_values.append(kl_divergence_from_nn(model=dbnn2).item())

        print(f"Train Accuracy: {round(train_accuracies_5[-1], 3)}%, Test Accuracy: {round(test_accuracies_5[-1], 3)}%")

    # Plotting accuracy and KL divergence over epochs
    plot_convergence(train_or_kl_list=train_accuracies_5, test_list=test_accuracies_5, epochs=epochs, mode='Accuracy',
                     model=5, question=3)
    plot_convergence(train_or_kl_list=model5_kl_values, test_list=[], epochs=epochs, mode='KL Divergence', model=5,
                     question=3)

    return dbnn2


@variational_estimator
class LogisticRegressionClassifier(nn.Module):
    """
    Logistic regression is pretty much a sigmoid function activated on a linear transformation.
    There's no need to use sigmoid function as it's already included in the CE Loss function.
    """

    def __init__(self, input_size, output_size):
        super(LogisticRegressionClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x_ = self.layer1(x)
        return x_


@variational_estimator
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
    models = []

    model_functions = ['model_1_train_and_test()', 'model_2_train_and_test()', 'model_3_train_and_test()',
                       'model_4_train_and_test()', 'model_5_train_and_test()']

    for i in range(len(model_functions)):
        model = eval(model_functions[i])
        models.append(model)

    # Save models to .pkl files
    save_models(models=models, question=3)

    # BEEP
    winsound.Beep(FREQ, DURATION)


if __name__ == '__main__':
    main()
