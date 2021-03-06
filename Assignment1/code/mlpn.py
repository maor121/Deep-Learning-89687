import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    layers = convert_params_to_layers(params)
    calculations = do_forward_prop(x, layers)
    probs = calculations[-1][2]
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):

    layers = convert_params_to_layers(params)

    num_labels = len(layers[-1][1])

    T = np.zeros(num_labels)
    T[y] = 1

    calculations = do_forward_prop(x, layers)
    grads = do_backprop(layers, calculations, T)

    loss = cost(calculations[-1][2], T)

    return [loss, grads]

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
            W = np.random.uniform(-1,1, (lastDim, nextDim))
            b = np.random.uniform(-1,1, nextDim)
            params += [W, b]

            lastDim = nextDim
        except StopIteration:
            break
    return params

############################# Helper functions ##############################

def pairs(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    from itertools import izip
    itr = iter(iterable)
    return izip(itr, itr)

def convert_params_to_layers(params):
    """Return a list of tuples [(W,b, is_last_layer: bool)...]"""
    layers = []
    for W, b in pairs(params):
        layers.append((W,b,False))
    last_layer = layers[-1]
    layers[-1] = (last_layer[0], last_layer[1], True)
    return layers

# Define the cost function
def cost(Y, T):
    return - np.multiply(T, np.log(Y)).sum()

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
    for layer, calculation in zip(reversed(layers), reversed(forward_prop)):
        is_last = layer[2]
        if is_last:
            Y = calculation[2]
            E = softmax_deriative(Y, T)
        else:
            Z = calculation[1]
            W = prev_layer[0]
            E = np.multiply(activation_deriative(Z) , np.dot(E, W.T))
        X = calculation[0]
        gW = np.squeeze(np.asarray(np.dot(np.matrix(X).T , np.matrix(E))))
        gb = E
        grads = [gW, gb] + grads
        prev_layer = layer
    return grads

def calc_layer_output(X, W, b):
    return X.dot(W) + b


if __name__ == '__main__':

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    params = create_classifier([8, 6, 4, 2])

    def _loss_and_grad(x, params, layerIndex, i):
        paramsClone = np.copy(params)
        paramsClone[layerIndex*2 + i] = x
        loss,grads = loss_and_gradients(np.array([1,2,3,4,5,6,7,8]),0,paramsClone)
        return loss,grads[layerIndex*2 + i]

    for _ in xrange(10):
        Wh1 = np.random.uniform(-1, 1, (8, 6))
        bh1 = np.random.uniform(-1, 1, 6)
        Wh2 = np.random.uniform(-1, 1, (6, 4))
        bh2 = np.random.uniform(-1, 1, 4)
        Wo = np.random.uniform(-1, 1, (4, 2))
        bo = np.random.uniform(-1, 1, 2)

        gradient_check(lambda x: _loss_and_grad(x, params, 2, 0), Wo)
        gradient_check(lambda x: _loss_and_grad(x, params, 2, 1), bo)
        gradient_check(lambda x : _loss_and_grad(x, params, 1, 0), Wh2)
        gradient_check(lambda x : _loss_and_grad(x, params, 1, 1), bh2)
        gradient_check(lambda x : _loss_and_grad(x, params, 0, 0), Wh1)
        gradient_check(lambda x : _loss_and_grad(x, params, 0, 1), bh1)
