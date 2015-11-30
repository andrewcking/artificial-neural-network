from helper import *
"""
Single Layer of Neurons (input, output and some potentially hidden layers)
    Generate arrays for: inputs, outputs, errors
    Generate matrix for: weights (connections between layers)
"""
class Layer:
    def __init__(self, id, layer_size, prev_layer_size, bias):
        #id for debugging
        self.id = id
        #number of neurons in the layer
        self.num_neurons = layer_size
        #bias
        self.bias_value = bias

        #create an input array as big as the number of neurons
        self.input = [0] * self.num_neurons
        #create output array as big as the number of neurons + bias
        self.output = [0] * (self.num_neurons + self.bias_value)
        #first value in output is the bias
        self.output[0] = self.bias_value
        #array for error values
        self.error = [0] * self.num_neurons
        #our layer will connect to the layer in front of it (previous layer) using a weight matrix
        self.weight = make_matrix(prev_layer_size + self.bias_value, self.num_neurons)
        #initialize weights for all values xy of matrix initialize them using the between helper function
        for x in range(len(self.weight)):
            for y in range(len(self.weight[x])):
                self.weight[x][y] = between(-1.0, 1.0)
