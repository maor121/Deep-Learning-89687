import utils
import numpy as np

def sort_by_len(input_tensor, labels_tensor):
    # Sort by size
    x = [(i, l) for i, l in sorted(zip(input_tensor, labels_tensor), key=lambda (i, l): len(i))]
    input_tensor, labels_tensor = zip(*x)
    return input_tensor, labels_tensor

class Generator(object):
    def __init__(self, input, labels, batch_size):
        self._input = input
        self._labels = labels
        self._batch_size = batch_size
        self.reset()
    def reset(self):
        self.current = 0
    def __iter__(self):
        return self
    def __next__(self):
        start = self.current * self._batch_size
        end =  (self.current+1) * self._batch_size
        if end > len(self._input):
            raise StopIteration
        sub_input = self._input[start:end]
        sub_labels = self._labels[start:end]
        max_seq_len = max([len(seq) for seq in sub_input])
        sub_input, sub_input = shuffle_input_labels(sub_input, sub_labels)


    @staticmethod
    def __shuffle_input_labels(sub_input, sub_labels):

        #np.random.permutation(10)
if __name__ == '__main__':
    W2I, T2I, input_tensor, labels_tensor = utils.load_dataset("../data/train")

    input_tensor, labels_tensor = sort_by_len(input_tensor, labels_tensor)



    print(0)