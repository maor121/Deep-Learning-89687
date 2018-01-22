import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

class SNLI_Tagger(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(SNLI_Tagger, self).__init__()


class Attention(nn.Module):
    def __init__(self, hidden_size, labels_count):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.labels_count = labels_count

        self.f_mlp = self._create_mlp(self.hidden_size, self.hidden_size)
        self.g_mlp = self._create_mlp(2 * self.hidden_size, self.hidden_size)
        self.h_mlp = self._create_mlp(2 * self.hidden_size, self.hidden_size)

        self.final_layer = nn.Linear(self.hidden_size, self.labels_count)

        self.log_prob = nn.LogSoftmax()

    def _create_mlp(self, in_dim, out_dim):
        # Paper defined all sub mlps to be of depth 2
        # Relu between layers, except the final_layer
        l = []
        l.append(nn.Dropout(p=0.15))
        l.append(nn.Linear(in_dim, out_dim))
        l.append(nn.ReLU())
        l.append(nn.Dropout(p=0.15))
        l.append(nn.Linear(out_dim, out_dim))
        l.append(nn.ReLU())
        return nn.Sequential(*l)  # * used to unpack list

    def forward(self, *input):
        """Implement intra sentence attention"""
        sources_lin, targets_lin = input
        source_sent_len = sources_lin.shape[1]
        targets_sent_len = targets_lin.shape[1] # Hypothesis

        '''attend'''

        f1 = self.f_mlp(sources_lin.view(-1, self.hidden_size))
        f2 = self.f_mlp(targets_lin.view(-1, self.hidden_size))

        f1 = f1.view(-1, source_sent_len, self.hidden_size)
        f2 = f2.view(-1, targets_sent_len, self.hidden_size)

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, targets_sent_len)).view(-1, source_sent_len, targets_sent_len)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, source_sent_len)).view(-1, targets_sent_len, source_sent_len)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sources_lin, torch.bmm(prob1, targets_lin)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat(
            (targets_lin, torch.bmm(prob2, sources_lin)), 2)
        # batch_size x len2 x (hidden_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))
        g1 = g1.view(-1, source_sent_len, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, targets_sent_len, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        h = self.final_linear(h)

        # print 'final layer'
        # print h.data

        log_prob = self.log_prob(h)

        return log_prob


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.encoder_linear = Linear(
            self.embedding_size, self.hidden_size)

    def forward(self, *input):
        sources, targets = input

        # reduce embedding dimesions to 200 as per the paper
        sources_lin = self.encoder_linear(sources)
        targets_lin = self.encoder_linear(targets)

        return sources_lin, targets_lin
