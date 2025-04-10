import numpy
import scipy.special
import matplotlib.pyplot as plt

# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,
                 activation_function = lambda x: scipy.special.expit(x)):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # weights info
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = activation_function

    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0-final_outputs)), numpy.transpose(hidden_errors))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs* (1.0-hidden_outputs)), numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        return self.activation_function(final_inputs)

def train_mnist():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()

    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for values in test_data_list:
        all_values = values.split(',')

        print(all_values[0])
        # 创建子图
        image_array = numpy.asarray(all_values[1:], dtype=float).reshape((28,28))
        fig, (ax1) = plt.subplots(1, 1)
        ax1.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
        print(n.query((numpy.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01))

def train_y_e_2x():
    input_nodes = 1
    hidden_nodes = 3
    output_nodes = 1
    learning_rate = 0.1

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, lambda x: x)
    for j in range(1000):
        i = j+1
        input = [i*0.0005]
        output = [i*0.001]
        n.train(input, output)

    for v in [0.3233, 0.2431, 0.3536534]:
        print('input: ', v, ", except answer: ", v*2, ", cal result: ", n.query([v]))


if __name__ == '__main__':
    train_mnist()