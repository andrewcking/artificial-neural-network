from helper import *
from layer import *
"""
Artificial Neural Network Class
"""
class NeuralNetwork:
    """Constructor"""
    def __init__(self, layer_sizes_array):
        self.neural_layers = []
        self.network_bias = 1
        self.learn_rate = 0.4
        #iterate through each element in the passed in layer_sizes array
        for x in range(len(layer_sizes_array)):
            #get element
            layer_size = layer_sizes_array[x]
            #if first layer 0 because no previous layer, otherwise get the element size of previous layer "layer_sizes[x-1]"
            prev_layer_size = 0 if x == 0 else layer_sizes_array[x - 1]
            #set up new layer
            new_layer = Layer(x, layer_size, prev_layer_size, self.network_bias)
            #add it to layers array
            self.neural_layers.append(new_layer)
    """Test the network using inputs and return outputs"""
    def run_test(self, input):
        self.set_inputs(input)
        self.forward_prop()
        return self.get_outputs()
    """Run a test and print a binary result, yes or no"""
    def run_bin_test(self, input_array):
        output = self.run_test(input_array)
        output_num = output[0]
        if output_num > 0.9:
            print("loan approved!")
        else:
            print("loan denied")
    """Set the input neuron outputs"""
    def set_inputs(self, input_array):
        #our input layer is just the first layer [0]
        input_layer = self.neural_layers[0]
        #set the outputs for the input layer equal to the specified inputs in the main class
        #also weird we don't need getters and setters... whoa python
        for i in range(0, input_layer.num_neurons):
            input_layer.output[i + self.network_bias] = input_array[i]
    """Sum the inputs for each neuron and propagate them forward"""
    def forward_prop(self):
        # exclude the last layer
        for x in range(len(self.neural_layers) - 1):
            # source layer
            source_layer = self.neural_layers[x]
            # destination layer
            dest_layer = self.neural_layers[x + 1]

            for j in range(0, dest_layer.num_neurons):
                sum_in = 0
                #sum all the inputs multiplied by their weights
                for i in range(0, source_layer.num_neurons + self.network_bias):
                    sum_in += dest_layer.weight[i][j] * source_layer.output[i]
                dest_layer.input[j] = sum_in
                #squash this using helper sigmoid function and set it as output of our destination layer
                dest_layer.output[j + self.network_bias] = sigmoid(sum_in)
    """Return array of outputs"""
    def get_outputs(self):
        output_layer = self.neural_layers[-1]
        #result
        result = [0] * output_layer.num_neurons
        for i in range(0, len(result)):
            result[i] = output_layer.output[i + self.network_bias]
        return result
    """Train the network with the labeled inputs for a max number of epochs(one forward pass and one backward pass of all the training examples)"""
    def train(self, inputs, targets, num_of_trains):
        #number of trains (epochs) means one forward pass and one backward pass over the network where the weights are adjusted once
        for epoch in range(0 , num_of_trains):
            #run forward with each input and backpropagate the error
            for i in range(0, len(inputs)):
                self.set_inputs(inputs[i])
                self.forward_prop()
                #difference between target and output
                self.update_error_output(targets[i])
                self.back_prop()
                self.update_weights()
    def update_error_output(self, target_array):
        output_layer = self.neural_layers[-1]
        for i in range(0, output_layer.num_neurons):
            neuron_output = output_layer.output[i + self.network_bias]
            #calculate the error (target-output)
            neuron_error = target_array[i] - neuron_output
            #because we are using the sigmoid activation function we have to get the derivative (where we fall on the curve)
            #to get the error we multiply that by our neuron error... thats the formula... not totally understanding why that works
            output_layer.error[i] = deriv_sigmoid(output_layer.input[i]) * neuron_error
    """Backpropagate the error from the output layer backwrds to the input layer"""
    def back_prop(self):
        #for each layer transfer (layers-1) backpropagate
        #this syntax means move back by 1 (-1) until we reach 0
        for x in range(len(self.neural_layers) - 1, 0, -1):
            source_layer = self.neural_layers[x]
            dest_layer = self.neural_layers[x - 1]
            for i in range(0, dest_layer.num_neurons):
                error = 0
                for j in range(0, source_layer.num_neurons):
                    error += source_layer.weight[i + self.network_bias][j] * source_layer.error[j]
                dest_layer.error[i] = deriv_sigmoid(dest_layer.input[i]) * error
    """update the weights matrix in each layer based on our error calculations"""
    def update_weights(self):
        for x in range(1, len(self.neural_layers)):
            for j in range(0, self.neural_layers[x].num_neurons):
                for i in range(0, self.neural_layers[x-1].num_neurons + self.network_bias):
                    out = self.neural_layers[x - 1].output[i]
                    error = self.neural_layers[x].error[j]
                    #this is the old weight + the learning rate * our error * the value of our output: (w--new = w--old + n * (t-y) * x
                    self.neural_layers[x].weight[i][j] += self.learn_rate * error * out
