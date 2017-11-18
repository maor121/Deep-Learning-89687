import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, layers):
    calculations = do_forward_prop(x, layers)
    probs = calculations[-1][2]
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, layers):

    num_labels = len(layers[-1][1])

    T = np.zeros(num_labels)
    T[y] = 1

    calculations = do_forward_prop(x, layers)
    grads = do_backprop(layers, calculations, T)

    return grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = []

    dimsIter = iter(dims)
    lastDim = dimsIter.next()
    while True:
        try:
            nextDim = dimsIter.next()
            W = np.random.uniform(0,1, (lastDim, nextDim))
            b = np.random.uniform(0,1, nextDim)
            params.append((W,b, False))

            lastDim = nextDim
        except StopIteration:
            lastLayer = params[-1]
            params[-1] = (lastLayer[0], lastLayer[1], True)
            break
    return params

############################# Helper functions ##############################

def softmax(x):
    shiftx = x - np.max(x)
    xExp = np.exp(shiftx)
    x = xExp / np.sum(xExp)
    return x

def activation(x):
    return np.tanh(x)

def softmax_deriative(Y, T):
    return Y - T

def activation_deriative(x):
    return 1 - (np.tanh(x) ** 2)

def do_forward_prop(X, layers):
    """Return [(X - input,Z - output before activation,A - output after activation) for each layer"""
    calculations = []
    for layer in layers:
        W, b, is_last_layer = layer
        Z = calc_layer_output(X, W, b)
        if is_last_layer:
            layer_out = softmax(Z)
        else:
            layer_out = activation(Z)
        calculations.append((X, Z, layer_out))
        X = layer_out
    return calculations

def do_backprop(layers, forward_prop, T):

    grads = []
    E = None
    prev_layer = None
    for layer, calculation in (reversed(layers), reversed(forward_prop)):
        is_last = layer[2]
        if is_last:
            Y = calculation[2]
            E = softmax_deriative(Y, T)
            prev_layer = layer
        else:
            Z = calculation[1]
            W = prev_layer[0]
            E = activation_deriative(Z) * (E * W.T)
        X = calculation[0]
        gW = X.T * E
        gH = E
        grads.append([gW, gH])

def calc_layer_output(x, W, b):
    return np.dot(x, W) + b
