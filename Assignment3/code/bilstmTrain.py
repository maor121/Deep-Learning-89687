import torch

import utils
from batchers import Generator

class BiLSTMTagger(torch.nn.Module):

    def __init__(self, repr_W, hidden_dim, target_size, is_cuda):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.repr_W = repr_W
        self.is_cuda = is_cuda
        self.target_size = target_size

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
    W2I, T2I, C2I, input_train, labels_train = utils.load_dataset("../data/pos/train", calc_characters=True)

    is_cuda = True
    learning_rate = 0.001
    batch_size = 20
    embedding_dim = 50
    hidden_dim = T2I.len() * 2
    vocab_size = W2I.len()
    num_chars = C2I.len()
    num_tags = T2I.len()
    epoches = 1

    import repr_w
    #repr_W = repr_w.repr_w_A_C(vocab_size, embedding_dim, is_cuda)
    repr_W = repr_w.repr_w_B(num_chars, embedding_dim/2, embedding_dim, is_cuda)
    #repr_W = repr_w.repr_w_D(vocab_size, num_chars, embedding_dim, embedding_dim, embedding_dim, embedding_dim, is_cuda)

    trainloader = Generator(input_train, labels_train, sort_dim=0, batch_size=batch_size)

    runner = BlistmRunner(learning_rate, is_cuda)
    runner.initialize_random(repr_W, hidden_dim, num_tags)
    runner.train(trainloader, epoches)

    # Eval
    __, __,__, input_test, labels_test = utils.load_dataset("../data/pos/dev", W2I=W2I, T2I=T2I, C2I=C2I, calc_characters=True)
    testloader = Generator(input_test, labels_test, sort_dim=0, batch_size=batch_size)
    runner.eval(testloader)

    print(0)