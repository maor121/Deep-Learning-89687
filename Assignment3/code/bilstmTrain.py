import torch
from torch.autograd import Variable
import torch.nn.functional as F

import utils
import numpy as np

def sort_by_len(input_tensor, labels_tensor):
    # Sort by size
    x = [(i, l) for i, l in reversed(sorted(zip(input_tensor, labels_tensor), key=lambda (i, l): len(i)))]
    input_tensor, labels_tensor = zip(*x)
    return input_tensor, labels_tensor

class Generator(object):
    def __init__(self, input, labels, batch_size):
        self._input = np.array(input, dtype=object)
        self._labels = np.array(labels, dtype=object)
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self.current = 0
        self._input, self._labels = self.__shuffle_input_labels(self._input, self._labels)

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
        sub_input, sub_labels = sort_by_len(sub_input, sub_labels)
        (input_tensor, lengths), labels_tensor = self.__build_tensors(sub_input, sub_labels)
        self.current += 1
        return (input_tensor, lengths), labels_tensor


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
        labels_tensor = torch.cat(sub_labels).long()
        lengths = []
        for i, (e,l) in enumerate(zip(sub_input,sub_labels)):
            length = len(e)
            input_tensor[i, :length] = e
            lengths.append(length)
        return (input_tensor, lengths), labels_tensor


class biLSTM(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(biLSTM, self).__init__()
        self.layers = np.ndarray(num_layers, dtype=object)
        self.num_layers = num_layers

        for i in range(num_layers):
            self.layers[i] = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, input, hx=None):
        b_input = input
        for i in range(self.num_layers):
            b_layer, __ = self.layers[i](b_input)
            #split b_f, b_b
            b_input = torch.cat(b_f,b_b)
        return b_input

class BiLSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size, is_cuda):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.is_cuda = is_cuda
        is_bidirectional = True
        lstm_num_layers = 2

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim,
                                  batch_first=True,
                                  bidirectional=is_bidirectional,
                                  num_layers=lstm_num_layers)

        # The linear layer that maps from hidden state space to tag space
        hidden_layer_in_dim = hidden_dim*2 if is_bidirectional else hidden_dim
        self.hidden_layer = torch.nn.Linear(hidden_layer_in_dim, hidden_layer_in_dim/2)
        self.out_layer = torch.nn.Linear(hidden_layer_in_dim/2, target_size)

    def forward(self, input):
        #print(self.word_embeddings.weight.data[0], self.word_embeddings.weight.data[1])
        #print(self.hidden_layer.weight[0])

        sentence, lengths = input

        a = Variable(sentence, volatile=not self.training)
        if self.is_cuda:
            a = a.cuda()
        max_seq_len = a.data.shape[1]
        words_depth = a.data.shape[2]

        b = a.view(-1, max_seq_len*words_depth)  # Unroll to (batch, seq_len*3)
        c = self.word_embeddings(b)  # To (batch, seq_len*3, embed_depth)
        d = c.view(-1, max_seq_len, words_depth, self.embedding_dim)  # Roll to (batch, seq_len, 3, 50)
        e = d.sum(2)  # Sum along 3rd axis -> (batch, seq_len, 50)

        pack = torch.nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True)
        lstm_out, __ = self.lstm(pack)

        #TODO: check if we need to concat output of lstm and reversed lstm

        #pack_padded_sequence changes order so it can batch, fix it back
        #pack[1,0] == e[0,1]
        lstm_out_unpacked, __ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out_chained = torch.cat([lstm_out_unpacked[batch, :l] for batch, l in enumerate(lengths)])

        hidden_out = F.tanh(self.hidden_layer(lstm_out_chained))
        out = self.out_layer(hidden_out)
        return out

from experiment import ModelRunner
class BlistmRunner(ModelRunner):
    def initialize_random(self, embedding_dim, hidden_dim, vocab_size, target_size):
        net = BiLSTMTagger(embedding_dim, hidden_dim, vocab_size, target_size, self.is_cuda)
        if (self.is_cuda):
            net.cuda()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net


if __name__ == '__main__':
    W2I, T2I, input_train, labels_train = utils.load_dataset("../data/train")
    __, __, input_test, labels_test = utils.load_dataset("../data/dev", W2I=W2I, T2I=T2I)

    is_cuda = True
    batch_size = 100
    learning_rate = 0.01
    embedding_dim = 50
    hidden_dim = T2I.len() * 2
    vocab_size = W2I.len()
    num_tags = T2I.len()
    epoches = 2

    trainloader = Generator(input_train, labels_train, batch_size)
    testloader = Generator(input_test, labels_test, batch_size)

    runner = BlistmRunner(learning_rate, is_cuda)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size, num_tags)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print(0)