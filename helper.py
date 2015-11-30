import random
import math

def make_matrix(X, Y):
    #Make an X rows by Y columns matrix
    return [[0 for i in range(Y)] for i in range(X)]
def between(min, max):
    #Return a real random value between min and max
    return random.random() * (max - min) + min
def sigmoid(x):
    #It takes an argument x and it squashes it to 0-1
    return 1.0 / (1 + math.exp(-x))
def deriv_sigmoid(x):
    sgmd = sigmoid(x)
    return (1 - sgmd) * sgmd
