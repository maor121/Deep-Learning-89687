import torch
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def flip_packed_sequence(pack):
    rev_var = pack.data.clone()
    start_batch = 0
    for batch_size in pack.batch_sizes:
        rev_var[start_batch:start_batch + batch_size] = flip(pack.data[start_batch:start_batch + batch_size], 0)
        start_batch += batch_size
    rev_lengths = list(pack.batch_sizes)
    return PackedSequence(rev_var, rev_lengths)


class BiLSTM(torch.nn.Module):
    """bilstm per the exercise definition"""
    def __init__(self, in_dim, out_dim, is_cuda):
        super(BiLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, out_dim, batch_first=True)
        self.revlstm = torch.nn.LSTM(in_dim, out_dim, batch_first=True)
        if is_cuda:
            self.lstm.cuda()
            self.revlstm.cuda()
            self.cuda()

    def forward(self, input):
        rev_input = flip_packed_sequence(input)

        #assert rev_input_var[input.batch_sizes[0]-1,0].data[0] == input.data[0,0].data[0]

        lstm_out, __ = self.lstm(input)
        revlstm_out, __ = self.revlstm(rev_input)

        #revlstm_out[0] is what lstm outputed from seeing last character (first for rev lstm)
        #flip again
        revlstm_out_flipped = flip_packed_sequence(revlstm_out)

        # b[i] = LSTM(x1,...xi) * LSTM(xn...xi)
        b = torch.cat([lstm_out.data, revlstm_out_flipped.data], 1)

        return PackedSequence(b, input.batch_sizes)


class MultLayerBiLSTM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, is_cuda):
        super(MultLayerBiLSTM, self).__init__()
        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(BiLSTM(in_dim, out_dim, is_cuda))
            in_dim = out_dim*2

        if is_cuda:
            self.cuda()
    def forward(self, input):
        return self.lstm_layers[0](input)