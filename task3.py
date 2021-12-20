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

# Define a softmax layer
def softmax(x):
    x = x.squeeze().astype(float)
    return np.array([np.exp(x) / sum(np.exp(x))])


# Define a single layer and its back propagation
# Shapes: x-[1, n] w-[n, m] b-[1, m]
def single_layer(x, w, b):
    # This is used to store the parameters for back propagation
    cache = (x, w, b)
    return cache, np.dot(x, w) + b


# The shape of the previous gradient shall be [1, m]
def single_layer_backward(prev_grad, cache):
    (x, w, b) = cache
    # Here we shall calculate the gradient of weight and bias
    grad_w = np.dot(x.transpose(), prev_grad)
    grad_b = np.sum(prev_grad, keepdims=True)
    current_grad = np.dot(prev_grad, w.transpose())
    grad = (grad_w, grad_b)
    return current_grad, grad


# We shall initialize the parameters based on the structure
def initializer(structure):
    weight = []
    bias = []

    # The total layer shall be n-1, where n denotes the dimension of the structure
    for i in range(len(structure) - 1):
        w = np.random.rand(structure[i], structure[i + 1]) / 10
        b = np.random.rand(1, structure[i + 1]) / 1000
        weight.append(w)
        bias.append(b)
    return weight, bias, len(structure) - 1


def cross_entropy(pred, y):
    return -sum(y * np.log(pred.squeeze()))



# The forward and the backward propagation of the network
def neural_network(x, y, layers, weight, bias, activation_type="relu", requires_cache=True):
    # This is used to store the parameters
    # When training, requires_cache is set to true
    caches = []
    temp = np.array(x, copy=True)
    for i in range(layers - 1):
        cache, temp = single_layer(temp, weight[i], bias[i])
        if requires_cache:
            caches.append(cache)

        # Apply the activation function
        if activation_type == "relu":
            temp = relu(temp)
        elif activation_type == "sigmoid":
            temp = sigmoid(temp)
        else:
            raise ValueError("No valid activation type")

    # The last layer is connected with softmax function
    cache, temp = single_layer(temp, weight[-1], bias[-1])
    if requires_cache:
        caches.append(cache)
    output = softmax(temp)
    if requires_cache:
        caches.append((output, y))

    # Use output to calculate the loss value
    loss = cross_entropy(output, y)

    # Return the predictions, the loss value and the caches
    return output, loss, caches


# The backward propagation of the network
# Caches store the variables in each layer and have the form of (x, w, b), despite of the last layer
def neural_network_backward(layers, caches, activation_type="relu"):
    # Use the last layer in the caches to initialize the gradient
    grads = []
    cache = caches[-1]
    (output, y) = cache
    current_grad = np.array([output.squeeze() - y])
    current_grad, grad = single_layer_backward(current_grad, caches[-2])
    grads.append(grad)

    # Deal with the rest of the layers
    for i in range(layers - 1):
        cache = caches[-i - 3]

        if activation_type == "relu":
            current_grad = relu_backward(current_grad, cache)
        elif activation_type == "sigmoid":
            current_grad = sigmoid_backward(current_grad, cache)
        else:
            raise ValueError("No valid activation type")

        # Backprop in layers
        current_grad, grad = single_layer_backward(current_grad, cache)

        # Save the gradient
        grads.append(grad)

    return grads
