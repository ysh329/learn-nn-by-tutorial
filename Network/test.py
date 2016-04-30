import Network.mnist_loader

data_dir = '../data/mnist.pkl.gz'
training_data, validation_data, test_data = \
    Network.mnist_loader.load_data_wrapper(data_dir)

print type(training_data)
print len(training_data)
print type(training_data[0])
print type(training_data[0][1])
print type(training_data[0][0][0])

print training_data[0][0][0]
print training_data[0][0][0][0]

print len(validation_data)

print len(test_data)

import network
net = network.Network([784, 30, 20, 20, 10])

net.SGD(training_data, 30, 50, 3.0, test_data=test_data)