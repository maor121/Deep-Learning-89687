import h5py
import torch
import numpy as np


def load_data(h5py_train_filename, h5py_test_filename, hdpy_word2vec_filename):
    with h5py.File(hdpy_word2vec_filename, 'r') as word2vec_file:
        w2v = dict_to_tensor(word2vec_file, 'word_vecs')

        vocab_size = len(w2v)
        embedding_size = len(w2v[0])

        w2v_emb = torch.nn.Embedding(vocab_size, embedding_size)
        w2v_emb.weight.data.copy_(w2v)
        w2v_emb.weight.requires_grad = False

        train_batches = load_batches(h5py_train_filename)
        test_batches = load_batches(h5py_test_filename)

        return train_batches, test_batches, w2v_emb


def dict_to_tensor(h5py_file, key_name, reduct_one=False):
    value = torch.from_numpy(np.array(h5py_file[key_name]))
    if reduct_one:
        value -= 1
    return value


def load_batches(h5py_filename):
    with h5py.File(h5py_filename, 'r') as h5py_file:

        sources = dict_to_tensor(h5py_file, 'source',reduct_one=True)
        # sources is zero padded, source_l is maximum length of the batch
        sources_lengths = dict_to_tensor(h5py_file, 'source_l')
        # Same for targets
        targets = dict_to_tensor(h5py_file, 'target',reduct_one=True)
        targets_lengths = dict_to_tensor(h5py_file, 'target_l')
        labels = dict_to_tensor(h5py_file, 'label', reduct_one=True)
        batches_lengths = dict_to_tensor(h5py_file, 'batch_l')
        batch_idx = dict_to_tensor(h5py_file, 'batch_idx', reduct_one=True)

        batches = [
            (sources[batch_start: batch_start+batch_len][:,:source_len],
             targets[batch_start: batch_start+batch_len][:,:target_len],
             labels[batch_start: batch_start+batch_len]
             )
            for batch_start,batch_len,source_len,target_len in
            zip(batch_idx, batches_lengths, sources_lengths, targets_lengths)
        ]

        return batches