import pickle
from blitz.losses import kl_divergence_from_nn


def evaluate():
    """
    Print the KL divergence value of each model.
    :return:
    """
    for i in range(5):
        model = pickle.load(open(f'./models/q2/q2_model_{i + 1}.pkl', 'rb'))
        model_kl_value = kl_divergence_from_nn(model)
        print(f'KL value of model {i + 1}: {model_kl_value}')


def main():
    evaluate()


if __name__ == '__main__':
    main()
