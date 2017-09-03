from network import Network
import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


if __name__ == '__main__':
    net = Network(3)
