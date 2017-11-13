import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b = params
    # YOUR CODE HERE.
    probs = nn(x, W[0], b[0], W[1], b[1])
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE

    out = predict(x, params)
    yOneHot = np.zeros(len(out))
    yOneHot[y] = 1

    loss = cost(out, yOneHot)

    W, b = params
    H = hidden_activations(x, W[0], b[1])

    Eo = error_output(out, yOneHot)
    gWo = gradient_weight_out(H, Eo)
    gbo = gradient_bias_out(Eo)

    Eh = error_hidden(H, W[1], Eo)
    gWh = gradient_weight_hidden(x, Eh)
    gbh = gradient_bias_hidden(Eh)

    return [loss, [[gWh, gWo],[gbh, gbo]]]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """

    Wh = np.random.uniform(0,1, (in_dim, hid_dim))
    bh = np.random.uniform(0,1, hid_dim)
    Wo = np.random.uniform(0,1, (hid_dim,out_dim))
    bo = np.random.uniform(0,1, out_dim)

    params = [[Wh, Wo],[bh, bo]]
    return params


########################## Helper functions #############################

# Define the softmax function
def softmax(x):
    shiftx = x - np.max(x)
    xExp = np.exp(shiftx)
    x = xExp / np.sum(xExp)

    return x

# Function to compute the hidden activations
def hidden_activations(X, Wh, bh):
    return np.tanh(X.dot(Wh) + bh)

# Define output layer feedforward
def output_activations(H, Wo, bo):
    return softmax(H.dot(Wo) + bo)

# Define the neural network function
def nn(X, Wh, bh, Wo, bo):
    return output_activations(hidden_activations(X, Wh, bh), Wo, bo)

# Define the cost function
def cost(Y, T):
    return - np.multiply(T, np.log(Y)).sum()

# Define the error function at the output
def error_output(Y, T):
    return Y - T

# Define the gradient function for the weight parameters at the output layer
def gradient_weight_out(H, Eo):
    return  H.T.dot(Eo)

# Define the gradient function for the bias parameters at the output layer
def gradient_bias_out(Eo):
    return  Eo

# Define the error function at the hidden layer
def error_hidden(H, Wo, Eo):
    # H * (1-H) * (E . Wo^T)
    return np.multiply(np.multiply(H,(1 - H)), Eo.dot(Wo.T))

# Define the gradient function for the weight parameters at the hidden layer
def gradient_weight_hidden(X, Eh):
    return X.T.dot(Eh)

# Define the gradient function for the bias parameters at the output layer
def gradient_bias_hidden(Eh):
    return  Eh