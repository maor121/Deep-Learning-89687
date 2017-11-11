import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    shiftx = x - np.max(x)
    xExp = np.exp(shiftx)
    x = xExp / np.sum(xExp)

    return x
    

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    # YOUR CODE HERE.
    probs = softmax(np.dot(x, W) + b)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

# Define the cost function
def cost(Y, T):
    return - np.multiply(T, np.log(Y)).sum()

# Define the error function at the output
def error_output(Y, T):
    return Y - T

# Define the gradient function for the weight parameters at the output layer
def gradient_weight_out(H, Eo):
    return np.matrix(H).T * np.matrix(Eo)

# Define the gradient function for the bias parameters at the output layer
def gradient_bias_out(Eo):
    return  np.sum(Eo, axis=0, keepdims=True)

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # YOU CODE HERE
    #Compute loss
    out = classifier_output(x,params=params)
    loss = -np.log10(out[y])

    yOneHot = np.zeros(len(out), dtype=np.int32)
    yOneHot[y] = 1

    # Backpropagation
    E0 = error_output(out, yOneHot)
    gW = gradient_weight_out(x, E0)
    gb = gradient_bias_out(E0)

    """"#Compute gW
    gW = np.ndarray(shape=W.shape)
    featureCount = len(out)
    yOneHot = np.zeros(featureCount)
    yOneHot[y] = 1
    out_signals = np.ndarray(len(out))
    # 1. compute output node signals
    for k in range(featureCount):
        derivative = (1 - out[k]) * out[k]
        out_signals[k] = derivative * (out[k] - yOneHot[k])
    # 2. compute input-to-output weight gradients using output signals
    for j in range(len(x)):
        for k in range(featureCount):
            gW[j, k] =  out_signals[k] * x[j]
    # 3. compute output node bias gradients using output signals
    #Compute gb
    gb = np.ndarray(shape=b.shape)
    for k in range(featureCount):
        gb[k] = out_signals[k] * 1.0"""

    return loss,[gW,gb]

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
