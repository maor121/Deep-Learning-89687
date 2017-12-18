import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMTagger, self).__init__()
        target_size = 2

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        out = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.unsqueeze(out[-1],0)


class ModelRunner:
    def __init__(self, learning_rate, is_cuda):
        self.learning_rate = learning_rate
        self.is_cuda = is_cuda
    def initialize_random(self, embedding_dim, hidden_dim, vocab_size):
        net = LSTMTagger(embedding_dim, hidden_dim, vocab_size)
        if (self.is_cuda):
            # from torch.backends import cudnn
            # cudnn.benchmark = True
            net.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net
    def train(self, trainloader, epoches):
        self.net.train(True)
        for epoch in range(epoches):  # loop over the dataset multiple times

            start_e_t = time.time()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                start_b_t = time.time()

                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                if self.is_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                self.net.init_hidden()
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                end_b_t = time.time()

                # print statistics
                running_loss += loss.data[0]
                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50, (end_b_t - start_b_t)))
                    running_loss = 0.0
            end_e_t = time.time()
            print('epoch time: %.3f' % (end_e_t - start_e_t))

    def eval(self, testloader):
        self.net.train(False)  # Disable dropout during eval mode
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            features, labels = data
            input = Variable(features, volatile=True)
            if self.is_cuda:
                input, labels = input.cuda(), labels.cuda()
            outputs = self.net(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            if i % 10 == 0 and i > 0:
                print("evaluated: "+str(i))
        print('Accuracy of the network on the %d test words: %d %%' % (
            total, 100 * correct / total))


def randomTrainingExample(C2I, ex_max_len):
    from gen_examples import gen_example
    example, is_positive = gen_example(ex_max_len)
    input_tensor = torch.LongTensor([C2I[c] for c in example])
    category_tensor = torch.LongTensor([is_positive])

    #return torch.unsqueeze(input_tensor, 0), torch.unsqueeze(category_tensor, 0)
    return input_tensor, category_tensor

class Generator(object):
    def __init__(self, len, C2I, ex_max_len, generateFunction):
        self._len = len
        self.current = 0
        self._C2I = C2I
        self._ex_max_len = ex_max_len
        self._gen_func = generateFunction

    def __iter__(self):
        return self

    def __len__(self):
        return self._len

    def __next__(self):
        self.current += 1
        if self.current >= self.__len__():
            raise StopIteration
        return self._gen_func(self._C2I, self._ex_max_len)

    next = __next__  # Python 2 compatibility


if __name__ == '__main__':
    import torch.utils.data

    is_cuda = False
    embedding_dim = 50
    hidden_dim = 20
    learning_rate = 0.001
    batch_size = 1
    epoches = 1

    vocab = "0123456789abcd"
    vocab_size = len(vocab)
    C2I = {}
    for c in vocab:
        C2I[c] = len(C2I)


    trainloader = Generator(2500, C2I, 200, randomTrainingExample)
    testloader = Generator(200, C2I, 10000, randomTrainingExample)

    runner = ModelRunner(learning_rate, is_cuda)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print('Finished Training')

