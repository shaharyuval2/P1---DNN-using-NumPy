import matplotlib.pyplot as plt
import numpy as np


def splitData(file_path):
    raw_data = np.loadtxt(file_path, delimiter=",")

    labels = raw_data[:, 0]
    examples = raw_data[:, 1:] / 255

    return labels, examples


def showImage(examples, index):
    plt.imshow(examples[index].reshape(28, 28), cmap="grey")
    plt.show()


def one_hot(labels, classes=10):
    return np.eye(classes)[labels.astype(int)].T
