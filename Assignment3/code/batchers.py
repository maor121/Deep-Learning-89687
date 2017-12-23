"""This file implements naive batching for Pytorch. It allows grouping together sequences of the same length, and dispatch
them in random order. e.g batch of sequences of size 10, batch of sequences of size 20, batch of sequences of size 10.."""

import torch

import utils
import numpy as np


class Generator(object):
    def __init__(self, input, labels, batch_size = 10, sort_dim=None, flattened_labels=False):
        self._input = input
        self._labels = labels
        self._flattened_labels = flattened_labels

        sorted_input, labels = utils.sort_by_len(input, labels, dim=sort_dim)
        batches = []
        batch_labels = []
        batch_inputs = []
        last_seq_size = len(sorted_input[0]) if sort_dim is None else len(sorted_input[0][sort_dim])
        for seq, lab in zip(sorted_input, labels):
            seq_len = len(seq) if sort_dim is None else len(seq[sort_dim])
            if seq_len != last_seq_size:
                batches.append((batch_inputs,batch_labels))
                batch_labels = [lab]
                batch_inputs = [seq]
                last_seq_size = seq_len
            else:
                batch_inputs.append(seq)
                batch_labels.append(lab)
        batches.append((batch_inputs, batch_labels))

        self.loaders = []
        for sub_input, sub_labels in batches:
            self.loaders.append(
                DataLoader(sub_input, sub_labels, batch_size=batch_size, flattened_labels=flattened_labels)
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
                if len(self.loader_iterators) == 0:
                    self.reset()
                    raise StopIteration

    next = __next__  # Python 2 compatibility


class DataLoader(object):
    def __init__(self, input, labels, batch_size, flattened_labels=False):
        # Convert labels to array
        self._labels = np.array(labels, dtype=object)

        # Convert input to array, manually because numpy likes to infer nested dimensions
        self._input = utils.list_to_array(input)

        self._flattened_labels = flattened_labels

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
        if self._flattened_labels:
            labels_tensor = sub_labels
        else:
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
