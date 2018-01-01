"""Usage: blistmTag.py <repr> <modelFile> <inputFile> [options]

-h --help    show this
-n           ner evaluation

"""
from docopt import docopt

import utils

if __name__ == '__main__':
    from bilstmTrain import Generator

    arguments = docopt(__doc__, version='Naval Fate 2.0')
    train_file = arguments['<inputFile>']
    model_file = arguments['<modelFile>']
    dev_file = arguments.get('<devFile>', None)
    repr = arguments['<repr>']


    legal_repr = ['a', 'b', 'c', 'd']
    if repr not in legal_repr:
        print("Illegal repr. Choose one of"+str(legal_repr))

    calc_sub_word = repr == 'c'
    calc_characters = repr in ['b', 'd']
    sort_dim = 0 if repr in ['b', 'd'] else None

    is_cuda = True

    runner = None
    W2I, T2I, F2I, C2I = None


    # Eval
    __, __, __, __, input_test, labels_test = utils.load_dataset(train_file, W2I=W2I, T2I=T2I, F2I=F2I, C2I=C2I,
                                                                 calc_characters=calc_characters,
                                                                 calc_sub_word=calc_sub_word)
    testloader = Generator(input_test, labels_test, batch_size=1000, sort_dim=sort_dim)

    omit_o_tag = T2I.get_id('O') if is_ner else False

    runner.predict(trainloader, epoches, testloader, omit_tag_id=omit_o_tag, plot=True)

    print(0)