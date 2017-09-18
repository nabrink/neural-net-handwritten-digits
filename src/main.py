import data_loader
from network import Network
import numpy as np




if __name__ == '__main__':
    training_data, validation_data, test_data = data_loader.load_data_wrapper()
    net = Network([784, 100, 10])
    net.SGD(training_data, 50, 20, 2.0, test_data=test_data)
