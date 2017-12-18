import torch

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
        if start >= len(self._input):
            raise StopIteration
        if end > len(self._input):
            end = len(self._input)
        sub_input = self._input[start:end]
        sub_labels = self._labels[start:end]
        sub_input, sub_labels = self.__shuffle_input_labels(sub_input, sub_labels)
        input_tensor, labels_tensor = self.__build_tensors(sub_input, sub_labels)
        return input_tensor, labels_tensor
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
        seq_depth = sub_input[0].shape(1)
        input_tensor = torch.zeros(batch_size, max_seq_len, seq_depth).long()
        labels_tensor = torch.LongTensor(sub_labels)
        for i, e in enumerate(sub_input):
            l = len(e)
            offset = max_seq_len - l
            input_tensor[i, offset:max_seq_len] = e
        return input_tensor, labels_tensor
if __name__ == '__main__':
    W2I, T2I, input_tensor, labels_tensor = utils.load_dataset("../data/train")

    input_tensor, labels_tensor = sort_by_len(input_tensor, labels_tensor)



    print(0)