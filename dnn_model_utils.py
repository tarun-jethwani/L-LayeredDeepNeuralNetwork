import numpy as np

def initialize_parameters(layers_dims):

    """
    Arguments :

    layer_dims --- python array (list) containing the dimensions of each layer in our network

    parameters --- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"

    Wl -->  weight matrix of shape (layer_dims[1], layer_dims[l-1])
    bl --> weight matrix of shape (layer_dims[l])

    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))


    return parameters

def linear_forward(A, W, b):

    """
    Arguments :
    A --- Activation from Previous Layer (or input data)
    W --- numpy matrix of shape (size of current layer, size of previous layer)
    b --- bias vector, numpy array of shape ( size of current layer, size of previous layer)

    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s, z


def relu(z):
    s = z * (z > 0)
    return np.abs(z), z


def linear_activation_forward(A_prev, W, b, activation):

    """
        Arguments

        A_prev --- activation from previous layer ( or input data): (size of previous layer, number of examples)
        W --- Weight matrix numpy array of shape (size of current_layer, size of previous_layer)
        b --- bias vector, numpy array of shape (size of the current layer, 1)
        activation --- the activation to be used stored as a text string: "sigmoid" or "relu"

        activation --- the ouput of the activation function, also called the post-activation value

        Returns:

        A --- the output of the activation function, also called the post-activation value
        cache --- A python dictionary containing "linear_cache" and "activaition_cache"

    """

    if activation == "sigmoid" :

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu' :

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):

    """
        Arguments :

        X --- data, numpy array of shape(input_size, number_of_examples)
        parameters --- initialized parameters

        Returns :

        AL --- Last post-activation value
        caches --- List of caches containing:
                    every cache of linear_relu_forward() (total L-1 caches are there, indexed from 0 to L-2)
                    cahce of linear_sigmoid_forward() ( there is one indexed L-1)

    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             activation = 'relu')

        caches.append(cache)



    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
                                          parameters['b' + str(L)],
                                          activation='sigmoid')

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))

    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ, cache):

    """

    Implement the linear portion of backward propogation for a single layer (layer l)

    Arguments :

        dZ --- Gradient of the cost wrt linear output of current layer
        cache --- tuple of values (A_prev, W, b) coming from forward propogation in the current layer

    Returns :

        dA_prev --- Gradient of the cost wrt activation (of the previous layer l-1), same shape as A_prev
        dW --- Gradient of the cost wrt W (current layer l), same shape as W,
        db --- Gradient of the cost wrt b ( current layer l), same shape as b

    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    dZ = dA * (activation_cache > 0)
    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s) * dA
    return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
        dA --- post-activation gradient for current layer
        cache --- tuple of values (linear_cache, activation_cache) we store for
        computing backward propogation efficiently
        activaition --- the activation to be used in this layer,
        stored as a text string: "sigmoid" or "relu"

    Returns:
        dA_prev --- Gradient of the cost with respect to the activation (of the previous layer l-1),
                same shape as A_prev
        dW --- Gradient of the cost with respect to W (current layer l), same shape as W
        db --- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)


    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):

    """
    Implement the backward propogation for the (LINEAR -> RELU) * (L-1) -> (LINEAR -> SIGMOID) group

    Arguments:

    AL --- probability vector, output of the forward propogation (L_model_forward())
    Y --- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing :
              every cache of linear_activation_forward() with 'relu' it's caches[1], for l in range(L-1)
              i.e l = 0 .... L-2)
              the cache of linear_activation_forward() with "sigmoid", its caches[L-1]

    Returns:

    grads --- A dictionary with the gradients
        grads["dA" + str(l)] = ...
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ...

    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Initializing the Back Propogation
    dAL = dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL,
                                                                                       current_cache[1]),
                                                                                       current_cache[0])

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(relu_backward(grads["dA" + str(l+2)],
                                                                       current_cache[1]),
                                                                       current_cache[0])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Arguments :
        parameters --- python dictionary containing parameters
        grads --- python dictionary containing gradients

    Returns :
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)]  = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)]  = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters
