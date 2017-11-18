import numpy as np
import mlp1
import random
import xor_data

def feats_to_vec(features):
    histogram = np.histogram(features, bins=2, range=(0, 1))[0] #Histogram, order does not matter
    return histogram

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features)
        prediction = mlp1.predict(x, params)
        if prediction == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            W,b = params

            Wh, Wo = W
            bh, bo = b
            np.subtract(Wo, learning_rate*grads[0][1], out=Wo)
            np.subtract(bo, learning_rate*grads[1][1], out=bo)
            np.subtract(Wh, learning_rate*grads[0][0], out=Wh)
            np.subtract(bh, learning_rate*grads[1][0], out=bh)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    in_dim = 2
    hid_dim = 2
    out_dim = 2
    train_data = xor_data.data
    dev_data = xor_data.data
    num_iterations = 330
    learning_rate = 0.08

    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
