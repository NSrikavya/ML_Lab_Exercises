import numpy as np
def summation_unit(inputs, weights, bias=0):
    return np.dot(inputs, weights) + bias
def step_function(x):
    return 1 if x >= 0 else 0
def bipolar_step_function(x):
    return 1 if x >= 0 else -1
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))
def tanh_function(x):
    return np.tanh(x)
def relu_function(x):
    return np.maximum(0, x)
def leaky_relu_function(x, alpha=0.01):
    return x if x > 0 else alpha * x
def comparator_unit(predicted, target):
    """
    predicted: output from perceptron
    target: actual desired output
    """
    error = target - predicted
    return error
inputs = np.array([1, 0])   
weights = np.array([0.5, 0.5])  
bias = -0.2

net_input = summation_unit(inputs, weights, bias)
print("Summation Output:", net_input)

activated_output = step_function(net_input)
print("Activated Output (Step):", activated_output)

target_output = 1
error = comparator_unit(activated_output, target_output)
print("Error:", error)
