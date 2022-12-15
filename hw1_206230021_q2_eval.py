import pickle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from hw1_206230021_q2_train import model_1_train_and_test, model_2_train_and_test, model_3_train_and_test, \
    model_4_train_and_test, model_5_train_and_test


def evaluate():
    model_1_train_and_test()
    model_2_train_and_test()
    model_3_train_and_test()
    model_4_train_and_test()
    model_5_train_and_test()


def main():
    evaluate()


if __name__ == '__main__':
    main()
