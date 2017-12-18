import torch
from torch.autograd import Variable

import utils
import numpy as np

def sort_by_len(input_tensor, labels_tensor):
    # Sort by size
    x = [(i, l) for i, l in sorted(zip(input_tensor, labels_tensor), key=lambda (i, l): len(i))]
    input_tensor, labels_tensor = zip(*x)
    return input_tensor, labels_tensor

class Generator(object):
    def __init__(self, input, labels, batch_size):
        self._input = np.array(input)
        self._labels = np.array(labels)
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        start = self.current * self._batch_size
        end =  (self.current+1) * self._batch_size
        if start >= len(self._input):
            raise StopIteration
        if end > len(self._input):
            end = len(self._input)
        sub_input = self._input[start:end]
        sub_labels = self._labels[start:end]
        sub_input, sub_labels = self.__shuffle_input_labels(sub_input, sub_labels)
        input_tensor, labels_tensor = self.__build_tensors(sub_input, sub_labels)
        return input_tensor, labels_tensor

    next = __next__  # Python 2 compatibility

    @staticmethod
    def __shuffle_input_labels(sub_input, sub_labels):
        num_examples = len(sub_input)
        permutations = np.random.permutation(num_examples)
        sub_input = sub_input[permutations]
        sub_labels = sub_labels[permutations]
        return sub_input, sub_labels

    @staticmethod
    def __build_tensors(sub_input, sub_labels):
        batch_size = len(sub_input)
        max_seq_len = max([len(seq) for seq in sub_input])
        seq_depth = sub_input[0].shape[1]
        input_tensor = torch.zeros(batch_size, max_seq_len, seq_depth).long()
        labels_tensor = torch.zeros(batch_size, max_seq_len)
        for i, (e,l) in enumerate(zip(sub_input,sub_labels)):
            length = len(e)
            offset = max_seq_len - length
            input_tensor[i, offset:max_seq_len] = e
            labels_tensor[i, offset:max_seq_len] = l
        return input_tensor, labels_tensor


class BiLSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(BiLSTMTagger, self).__init__()
        target_size = 2

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, target_size)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                       torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        a = sentence
        max_seq_len = a.data.shape[1]
        words_depth = a.data.shape[2]
        b = a.view(-1, max_seq_len*words_depth)  # Unroll to (batch, seq_len*3)
        #c = self.word_embeddings.weight.data.numpy().take(b.data.numpy(), 0)
        #c = Variable(torch.FloatTensor(c), volatile = not self.training)
        c = self.word_embeddings(b)  # To (batch, seq_len*3, embed_depth)
        d = c.view(-1, max_seq_len, words_depth, self.embedding_dim)  # Roll to (batch, seq_len, 3, 50)
        e = d.sum(2)  # Sum along 3rd axis -> (batch, seq_len, 50)
        #x = e.view(-1, self.embedding_dim * max_seq_len)

        lstm_out, self.hidden = self.lstm(
            e.view(-1, max_seq_len, words_depth), self.hidden)
        # Remove padding

        out = self.hidden2tag(lstm_out.view(-1, max_seq_len))
        return torch.unsqueeze(out[-1],0)

from experiment import ModelRunner
class BlistmRunner(ModelRunner):
    def initialize_random(self, embedding_dim, hidden_dim, vocab_size):
        net = BiLSTMTagger(embedding_dim, hidden_dim, vocab_size)
        if (self.is_cuda):
            # from torch.backends import cudnn
            # cudnn.benchmark = True
            net.cuda()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net


if __name__ == '__main__':
    W2I, T2I, input_list, labels_list = utils.load_dataset("../data/train")

    input_list, labels_list = sort_by_len(input_list, labels_list)

    is_cuda = False
    batch_size = 128
    learning_rate = 0.001
    embedding_dim = 50
    hidden_dim = T2I.len() * 4
    vocab_size = W2I.len()
    epoches = 1

    trainloader = Generator(input_list, labels_list, batch_size)

    runner = BlistmRunner(learning_rate, is_cuda)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    print(0)