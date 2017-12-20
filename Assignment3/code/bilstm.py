import torch
import numpy as np


class BiLSTM(torch.nn.Module):
    """bilstm per the exercise definition"""
    def __init__(self, in_dim, out_dim):
        self.lstm = torch.nn.LSTM(in_dim, out_dim, batch_first=True)
        self.revlstm = torch.nn.LSTM(in_dim, out_dim, batch_first=True)

    def forward(self, input):
        lstm_out, __ = self.lstm(input)
        revlstm_out, __ = self.revlstm(input)

        #TODO: debug this
        b = [torch.cat([]) for out_i, revout_i in zip(lstm_out)]

        return b


class MultLayerBiLSTM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        self.lstm_layers = np.ndarray(num_layers, dtype=object)
        for i in range(num_layers):
            self.lstm_layers[i] = BiLSTM(in_dim, out_dim)
    def forward(self, input):
        for bilstm in self.lstm_layers:
            #TODO: debug this
            input = bilstm(input)
        return input