import loglinear as ll
import random
import numpy as np
import utils

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    vocab_size = utils.VOCABOLARY_SIZE
    histogram = np.histogram(features, bins=vocab_size, range=(0,vocab_size-1))[0] # Drop UNK word (id=600)
    normalized_histogram = histogram / float(len(features)) #normalize histogram by vector length
    return normalized_histogram

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)
        prediction = ll.predict(x, params)
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
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            W,b = params
            np.subtract(W, learning_rate*grads[0], out=W)
            np.subtract(b, learning_rate*grads[1], out=b)

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    TRAIN, DEV, L2I, F2I = utils.read_dataset("../data/train", "../data/dev", ngrams=2)

    in_dim = len(F2I)
    out_dim = len(L2I)
    train_data = utils.dataset_to_ids(TRAIN, F2I, L2I)
    dev_data = utils.dataset_to_ids(DEV, F2I, L2I)
    num_iterations = 50
    learning_rate = 1

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

