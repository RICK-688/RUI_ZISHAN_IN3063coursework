import numpy as np



# Here we define the activation functions and their back propagation
# No need to preserve parameters for these calculations can be done based on these values
def sigmoid(x):
    return np.array([1 / (1 + np.exp(-x.squeeze()))])


def relu(x):
    return np.array([np.maximum(0, x.squeeze())])


# The shape of the previous gradient shall be [1, n]
def sigmoid_backward(prev_grad, cache):
    (x, w, b) = cache
    temp = np.dot(x, w) + b
    return np.array([prev_grad.squeeze() * sigmoid(temp).squeeze() * (1 - sigmoid(temp).squeeze())])


def relu_backward(prev_grad, cache):
    (x, w, b) = cache
    temp = np.dot(x, w) + b
    output = []
    for i in range(len(temp.squeeze())):
        if temp.squeeze()[i] > 0:
            output.append(prev_grad.squeeze()[i])
        else:
            output.append(0)
    return np.array([output])