"""Usage: blistmTrain.py <repr> <trainFile> <modelFile> [<devFile>] [--ner] [--plot]

-h --help    show this
--ner        ner evaluation mode
--plot       plot accuracy graph at the end, requires <devFile> to exist

"""
from docopt import docopt

import torch

import utils
import pickle
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

    def forward(self, input):
        e = self.repr_W(input)

        lstm_out, __ = self.bilstm(e)

        out = self.out_layer(lstm_out)
        return out.view(-1, self.target_size) # unroll to a long vector

from experiment import ModelRunner
class BlistmRunner(ModelRunner):
    def initialize_random(self, repr_W, hidden_dim, target_size):
        net = BiLSTMTagger(repr_W, hidden_dim, target_size, self.is_cuda)
        if self.is_cuda:
            net.cuda()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net


if __name__ == '__main__':
    import repr_w

    arguments = docopt(__doc__, version='Naval Fate 2.0')
    train_file = arguments['<trainFile>']
    model_file = arguments['<modelFile>']
    dev_file = arguments.get('<devFile>', None)
    repr = arguments['<repr>']
    is_ner = arguments['--ner']
    is_plot = arguments['--plot']


    legal_repr = ['a', 'b', 'c', 'd']
    if repr not in legal_repr:
        print("Illegal repr. Choose one of"+str(legal_repr))
        exit()
    if is_plot and not dev_file:
        print("To set --plot you must supply a dev file")
        exit()

    calc_sub_word = repr == 'c'
    calc_characters = repr in ['b', 'd']
    sort_dim = 0 if repr in ['b', 'd'] else None

    W2I, T2I, F2I, C2I, input_train, labels_train = utils.load_dataset(train_file, calc_sub_word=calc_sub_word, calc_characters=calc_characters)

    is_cuda = True
    learning_rate = 0.001
    batch_size = 20
    embedding_dim = 50
    hidden_dim = T2I.len() * 4
    vocab_size = W2I.len()
    num_tags = T2I.len()
    epoches = 5

    if repr == 'a':
        repr_W = repr_w.repr_w_A_C(vocab_size, embedding_dim, is_cuda)
    else:
        if repr == 'c':
            repr_W = repr_w.repr_w_A_C(vocab_size+F2I.len(),embedding_dim, is_cuda)
        else:
            num_chars = C2I.len()
            if repr == 'b':
                repr_W = repr_w.repr_w_B(num_chars, embedding_dim/2, embedding_dim, is_cuda)
            else:
                #d
                repr_W = repr_w.repr_w_D(vocab_size, num_chars, embedding_dim, embedding_dim, embedding_dim, embedding_dim, is_cuda)

    trainloader = Generator(input_train, labels_train, batch_size=batch_size, sort_dim=sort_dim)

    # Eval
    if dev_file is not None:
        __, __, __, __, input_test, labels_test = utils.load_dataset(dev_file, W2I=W2I, T2I=T2I, F2I=F2I, C2I=C2I,
                                                                     calc_characters=calc_characters,
                                                                     calc_sub_word=calc_sub_word)
        testloader = Generator(input_test, labels_test, batch_size=1000, sort_dim=sort_dim)

        omit_o_tag = T2I.get_id('O') if is_ner else False
    else:
        omit_o_tag = testloader = None

    runner = BlistmRunner(learning_rate, is_cuda, 500)
    runner.initialize_random(repr_W, hidden_dim, num_tags)
    runner.train(trainloader, epoches, testloader, omit_tag_id=omit_o_tag, plot=is_plot)

    if dev_file is not None:
        runner.eval(testloader, omit_tag_id=omit_o_tag, to_print=True)

    print("Saving model... (If it's POS, this may take a while)")
    pickle.dump((W2I, T2I, F2I, C2I, runner), open(model_file, 'w+'))

    print(0)