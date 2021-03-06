import time
import torch
from torch import nn
from torch.autograd import Variable
import random

from model import SNLI_Tagger


def convert_batch_to_embedding(batch, w2v):
    embeddings = w2v(batch)
    return embeddings


def penalize_sent_len(train_loss, dev_loss):
    global last_train_loss, last_dev_loss, dropout_rate
    if 'last_train_loss' not in globals():
        last_train_loss = 999
    if 'last_dev_loss' not in globals():
        last_dev_loss = 999
    if 'dropout_rate' not in globals():
        dropout_rate = 0

    train_momentum = last_train_loss - train_loss
    dev_momentum = last_dev_loss - dev_loss
    last_train_loss = train_loss
    last_dev_loss = dev_loss

    sent_len_penalty = 0
    dropout_rate = dev_loss - train_loss
    if train_momentum > dev_momentum * 10:
        print("Possible overfitting: max_sent_len++")
        sent_len_penalty = 1
    return sent_len_penalty, dropout_rate


def get_n_params(model):
    """Count parameters of a model"""
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class ModelRunner:
    def __init__(self, learning_rate, weight_decay, lr_decay, is_cuda):
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.is_cuda = is_cuda
    def initialize_random(self, w2v, hidden_size, labels_count):
        self.w2v = w2v

        embedding_size = w2v.weight.data.shape[1]

        self.net = SNLI_Tagger(embedding_size, hidden_size, labels_count)
        if (self.is_cuda):
            self.net = self.net.cuda()
            self.w2v = self.w2v.cuda()

        self.criterion = nn.CrossEntropyLoss(size_average=True)

        self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, lr_decay=lr_decay)

    def train(self, trainloader, epoches, testloader, initial_max_sent_len=16):
        print '=' * 30
        print "Begin training of model"
        print str(get_n_params(self.net)) + ' parameters'
        print '=' * 30

        max_sent_len = initial_max_sent_len
        train_dropout = 0

        self.net.train(True)
        for epoch in range(epoches):

            # Shuffle train batches
            random.shuffle(trainloader)

            start_e_t = time.time()
            running_loss = 0.0
            sent_trained_in_epoch = 0
            for i, data in enumerate(trainloader, 0):
                start_b_t = time.time()

                # get the inputs
                sources, targets, labels = data

                src_len = sources.shape[1]
                trg_len = targets.shape[1]
                batch_size = sources.shape[0]
                if batch_size == 1:
                    continue # Skip batches of size 1, gives error for BatchNorm1d
                if src_len > max_sent_len or trg_len > max_sent_len:
                    continue
                #if random.uniform(0,1) < 0.2:
                #    continue # dropout training data to avoid overfitting

                # Wrap tensors in variables
                sources = Variable(sources)
                targets = Variable(targets)
                labels = Variable(labels)

                if self.is_cuda:
                    sources = sources.cuda()
                    targets = targets.cuda()
                    labels = labels.cuda()

                # Convert to embeddings
                sources = convert_batch_to_embedding(sources, self.w2v)
                targets = convert_batch_to_embedding(targets, self.w2v)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net((sources, targets))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                end_b_t = time.time()

                # print statistics
                running_loss += loss.data[0]

                sent_trained_in_epoch += 1

            train_loss, train_acc = self.eval(trainloader)
            dev_loss, dev_acc = self.eval(testloader)

            end_e_t = time.time()

            print('[%d, %5d] epoch time: %.3f\ntrain_loss: %.3f, train_acc: %.3f\ndev_loss: %.3f, dev_acc: %.3f' %
                  (epoch + 1, sent_trained_in_epoch + 1,
                   (end_e_t - start_e_t),
                   train_loss, train_acc,
                   dev_loss, dev_acc
                   ))
            #pen_sent_len, train_dropout = penalize_sent_len(train_loss, dev_loss)
            #max_sent_len = max_sent_len + pen_sent_len

    def eval(self, testloader):
        self.net.train(False)  # Disable dropout during eval mode
        correct = 0
        total = 0
        total_loss = 0.0
        for i, data in enumerate(testloader):
            sources, targets, labels = data
            sources = Variable(sources)
            targets = Variable(targets)
            labels = Variable(labels)
            if self.is_cuda:
                labels = labels.cuda()
                sources = sources.cuda()
                targets = targets.cuda()
            sources = convert_batch_to_embedding(sources, self.w2v)
            targets = convert_batch_to_embedding(targets, self.w2v)
            outputs = self.net((sources, targets))
            loss = self.criterion(outputs, labels)
            total_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        return total_loss / len(testloader), 100.0 * correct / total


if __name__ == '__main__':
    import utils
    import sys
    if len(sys.argv) != 4:
        print 'Wrong num of arguments. Try:\npython train.py train.hdf5 dev.hdf5 glove300d.hdf5'

    train_filename = sys.argv[1]
    dev_filename = sys.argv[2]
    w2v_filename = sys.argv[3]

    train_batches, test_batches, w2v = utils.load_data(
        train_filename, dev_filename, w2v_filename
    )

    vocab_size = w2v.weight.data.shape[0]
    lr = 0.01
    weight_decay = 1e-6
    lr_decay = 0
    hidden_size = 300
    epoches = 17
    labels_count = 3
    is_cuda = True

    model_runner = ModelRunner(lr, weight_decay, lr_decay, is_cuda)
    model_runner.initialize_random(w2v, hidden_size, labels_count)
    model_runner.train(train_batches, epoches, test_batches)

    print(0)