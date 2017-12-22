import torch
from torch.utils.data import TensorDataset

import utils
import numpy as np


class Generator(object):
    def __init__(self, input, labels, batch_size = 3):
        self._input = input
        self._labels = labels

        sorted_input, labels = utils.sort_by_len(input, labels, dim=0)
        batches = []
        batch_labels = []
        batch_inputs = []
        last_seq_size = len(sorted_input[0][0])
        for seq, lab in zip(sorted_input, labels):
            seq_len = len(seq[0])
            if seq_len != last_seq_size:
                batches.append((batch_inputs,batch_labels))
                batch_labels = []
                batch_inputs = []
                last_seq_size = seq_len
            else:
                batch_inputs.append(seq)
                batch_labels.append(lab)
        self.loaders = []
        for sub_input, sub_labels in batches:
            self.loaders.append(
                DataLoader(sub_input, sub_labels, batch_size=batch_size)
            )

        self.reset()

    def reset(self):
        self.loader_iterators = [loader.__iter__() for loader in self.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        import random

        if len(self.loader_iterators) == 0:
            self.reset()
            raise StopIteration

        while True:
            r_loader = random.choice(self.loader_iterators)
            try:
                next_batch = r_loader.next()
                return next_batch
            except StopIteration:
                self.loader_iterators.remove(r_loader)

    next = __next__  # Python 2 compatibility


class DataLoader(object):
    def __init__(self, input, labels, batch_size):
        self._input = np.array(input, dtype=object)
        self._labels = np.array(labels, dtype=object)

        self.current = 0
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._input, self._labels = self.__shuffle_input_labels(self._input, self._labels)
        self.current = 0
    def __iter__(self):
        return self
    def __next__(self):
        start = self.current * self._batch_size
        end =  (self.current+1) * self._batch_size
        if start >= len(self._input):
            self.reset()
            raise StopIteration
        if end > len(self._input):
            end = len(self._input)
        sub_input = self._input[start:end]
        sub_labels = self._labels[start:end]
        labels_tensor = torch.cat(sub_labels)
        self.current += 1
        return sub_input, labels_tensor

    next = __next__  # Python 2 compatibility

    @staticmethod
    def __shuffle_input_labels(sub_input, sub_labels):
        num_examples = len(sub_input)
        permutations = np.random.permutation(num_examples)
        sub_input = sub_input[permutations]
        sub_labels = sub_labels[permutations]
        return sub_input, sub_labels

class BiLSTMTagger(torch.nn.Module):

    def __init__(self, repr_W, hidden_dim, target_size, is_cuda):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.repr_W = repr_W
        self.is_cuda = is_cuda
        self.target_size = target_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #self.bilstm = bilstm.MultLayerBiLSTM(self.repr_W._embedding_dim, hidden_dim, 1, is_cuda)
        self.bilstm = torch.nn.LSTM(self.repr_W.out_dim(), hidden_dim,
                                    batch_first=True,
                                    bidirectional=True,
                                    num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        hidden_layer_in_dim = hidden_dim*2
        self.out_layer = torch.nn.Linear(hidden_layer_in_dim, target_size)

        if is_cuda:
            self.cuda()

    def forward(self, input):
        e = self.repr_W(input)

        lstm_out, __ = self.bilstm(e)

        out = self.out_layer(lstm_out)
        return out.view(-1, self.target_size) # unroll to a long vector

from experiment import ModelRunner
class BlistmRunner(ModelRunner):
    def initialize_random(self, repr_W, hidden_dim, target_size):
        net = BiLSTMTagger(repr_W, hidden_dim, target_size, self.is_cuda)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net


if __name__ == '__main__':
    W2I, T2I, C2I, input_train, labels_train = utils.load_dataset("../data/train", calc_characters=True)

    is_cuda = True
    learning_rate = 0.001
    embedding_dim = 10
    hidden_dim = T2I.len() * 2
    vocab_size = W2I.len()
    num_chars = C2I.len()
    num_tags = T2I.len()
    epoches = 1

    import repr_w
    #repr_W = repr_w.repr_w_A_C(vocab_size, embedding_dim, is_cuda)
    #repr_W = repr_w.repr_w_B(num_chars, embedding_dim, embedding_dim, is_cuda)
    repr_W = repr_w.repr_w_D(vocab_size, num_chars, embedding_dim, embedding_dim, embedding_dim, embedding_dim, is_cuda)

    trainloader = Generator(input_train, labels_train)

    runner = BlistmRunner(learning_rate, is_cuda)
    runner.initialize_random(repr_W, hidden_dim, num_tags)
    runner.train(trainloader, epoches)

    # Eval
    __, __, input_test, labels_test = utils.load_dataset("../data/dev", W2I=W2I, T2I=T2I, C2I=C2I, calc_characters=True)
    testloader = Generator(input_test, labels_test)
    runner.eval(testloader)

    print(0)