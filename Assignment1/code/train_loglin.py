import loglinear as ll
import random
import numpy as np
import utils

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

global VOCABOLARY_SIZE

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    histogram = np.histogram(features, bins=VOCABOLARY_SIZE, range=(0,VOCABOLARY_SIZE-1))[0] # Drop UNK word (id=-1)
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

    VOCABOLARY_SIZE = 600
    NGRAMS = 2
    TRAIN, DEV, L2I, F2I = utils.read_dataset("../data/train", "../data/dev", ngrams=NGRAMS, vocab_size=VOCABOLARY_SIZE)

    in_dim = len(F2I)
    out_dim = len(L2I)
    train_data = utils.dataset_to_ids(TRAIN, F2I, L2I)
    dev_data = utils.dataset_to_ids(DEV, F2I, L2I)
    num_iterations = 50
    learning_rate = 1

    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    """
    I2L = {v: k for k, v in L2I.iteritems()}

    TEST = [(l, utils.text_to_ngrams(t, NGRAMS)) for l, t in utils.read_data("../data/test")]

    test_data = [[[utils.ngram_to_id(b, F2I) for b in blist]] for l, blist in iter(TEST)]
    for features in test_data:
        print(I2L[ll.predict(feats_to_vec(features), trained_params)])
    """