import mlp1
import random
import numpy as np

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    histogram = np.histogram(features, bins=600, range=(0,599))[0] # Drop UNK word (id=600)
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
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    #Load data on import
    import utils
    in_dim = len(utils.F2I)
    hid_dim = 32
    out_dim = len(utils.L2I)
    train_data = utils.dataset_to_ids(utils.TRAIN, utils.F2I, utils.L2I)
    dev_data = utils.dataset_to_ids(utils.DEV, utils.F2I, utils.L2I)
    num_iterations = 50
    learning_rate = 0.05

    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
