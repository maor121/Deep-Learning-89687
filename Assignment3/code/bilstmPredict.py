"""Usage: blistmPerdict.py <repr> <modelFile> <inputFile> <outputFile>

-h --help    show this
-n           ner evaluation

"""
from docopt import docopt

import utils
from bilstmTrain import BlistmRunner, BiLSTMTagger # For pickle
from utils import StringCounter # For pickle


class OrderedGenerator(object):
    def __init__(self, input, labels):
        self._input = input
        self._labels = labels

        self.reset()

    def reset(self):
        self.loader_iterators = [iter(self._input), iter(self._labels)]

    def __iter__(self):
        return self

    def __next__(self):

        while True:
            try:
                input = self.loader_iterators[0].next()
                label = self.loader_iterators[1].next()
                return [input], label
            except StopIteration:
                self.reset()
                raise StopIteration

    next = __next__  # Python 2 compatibility


if __name__ == '__main__':
    import pickle

    arguments = docopt(__doc__, version='Naval Fate 2.0')
    input_file = arguments['<inputFile>']
    model_file = arguments['<modelFile>']
    output_file = arguments['<outputFile>']
    repr = arguments['<repr>']

    legal_repr = ['a', 'b', 'c', 'd']
    if repr not in legal_repr:
        print("Illegal repr. Choose one of"+str(legal_repr))
        exit()

    calc_sub_word = repr == 'c'
    calc_characters = repr in ['b', 'd']
    sort_dim = 0 if repr in ['b', 'd'] else None

    is_cuda = True

    W2I, T2I, F2I, C2I, runner = pickle.load(open(model_file, 'r'))


    # Eval
    __, __, __, __, input_test, labels_test = utils.load_dataset(input_file, W2I=W2I, T2I=T2I, F2I=F2I, C2I=C2I,
                                                                 calc_characters=calc_characters,
                                                                 calc_sub_word=calc_sub_word)
    testloader = OrderedGenerator(input_test, labels_test)

    runner.write_prediction(testloader, T2I, input_file, output_file)

    print(0)