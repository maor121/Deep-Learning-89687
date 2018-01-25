import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear


class SNLI_Tagger(nn.Module):
    def __init__(self, embedding_size, hidden_size, labels_count):
        super(SNLI_Tagger, self).__init__()

        encoder_net = Encode(embedding_size, hidden_size)
        #intra_attention = IntraAttend(hidden_size)
        attention = Attend(hidden_size)
        compare = Compare(hidden_size)
        aggregate = Aggregate(hidden_size, labels_count)

        #self.net = Sequential(*[encoder_net, intra_attention, attention, compare, aggregate])
        self.net = nn.Sequential(*[encoder_net, attention, compare, aggregate])

    def forward(self, *input):
        return self.net(input[0])


class Aggregate(nn.Module):
    """Paper: Section 3.3"""
    def __init__(self, hidden_size, labels_count):
        super(Aggregate, self).__init__()

        self.hidden_size = hidden_size
        self.labels_count = labels_count

        h_mlp = create_2_linear_batchnorm_relu_dropout(2 * self.hidden_size, self.hidden_size)
        final_layer = nn.Linear(self.hidden_size, self.labels_count)

        self.H = nn.Sequential(*[h_mlp, final_layer])

    def forward(self, *input):
        v1i, v2j = input[0]

        v1 = torch.squeeze(torch.sum(v1i, 1), 1)
        v2 = torch.squeeze(torch.sum(v2j, 1), 1)

        y_kova = self.H(torch.cat((v1, v2), 1))

        return y_kova


class Compare(nn.Module):
    """Paper: Section 3.2"""
    def __init__(self, hidden_size):
        super(Compare,self).__init__()

        self.hidden_size = hidden_size
        self.G = create_2_linear_batchnorm_relu_dropout(2 * self.hidden_size, self.hidden_size)

    def forward(self, *input):
        alpha, beta = input[0]

        source_sent_len = alpha.shape[1]
        targets_sent_len = beta.shape[1]  # Hypothesis

        v1i = self.G(alpha.view(-1, 2 * self.hidden_size)).view(-1, source_sent_len, self.hidden_size)
        v2j = self.G(beta.view(-1, 2 * self.hidden_size)).view(-1, targets_sent_len, self.hidden_size)

        return v1i, v2j


class Attend(nn.Module):
    """Paper: Section 3.1"""
    def __init__(self, hidden_size):
        super(Attend, self).__init__()
        self.hidden_size = hidden_size

        self.f_mlp = create_2_linear_batchnorm_relu_dropout(hidden_size, hidden_size)

    def _forward_F_tag(self, a, b):
        a_sent_len = a.shape[1]
        b_sent_len = b.shape[1]
        a_f_mlp = self.f_mlp(a.view(-1, self.hidden_size)).view(-1, a_sent_len, self.hidden_size)
        b_f_mlp = self.f_mlp(b.view(-1, self.hidden_size)).view(-1, b_sent_len, self.hidden_size)

        return torch.bmm(a_f_mlp, torch.transpose(b_f_mlp, 1, 2))

    def forward(self, *input):
        sources_lin, targets_lin =input[0]
        source_sent_len = sources_lin.shape[1]
        targets_sent_len = targets_lin.shape[1]  # Hypothesis

        # Attention weights
        Eij = self._forward_F_tag(sources_lin, targets_lin)
        Eji = torch.transpose(Eij.contiguous(), 1, 2).contiguous() # transpose(Eij)

        # Normalize attention weights
        Eij_softmax = F.softmax(Eij.view(-1, targets_sent_len), dim=1).view(-1, source_sent_len, targets_sent_len)
        Eji_softmax = F.softmax(Eji.view(-1, source_sent_len), dim=1).view(-1, targets_sent_len, source_sent_len)

        #Attention part!

        # beta = subphrase of target which is softly aligned to source (source: paper)
        beta = torch.cat((targets_lin, torch.bmm(Eji_softmax, sources_lin)), 2)
        # alpha = subphrase of source which is softly aligned to target (source: paper)
        alpha = torch.cat((sources_lin, torch.bmm(Eij_softmax, targets_lin)), 2)

        return alpha, beta

class Encode(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Encode, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        l = []
        l.append(Linear(self.embedding_size, self.hidden_size))
        l.append(ReshapeLayer(-1, self.hidden_size))
        l.append(nn.BatchNorm1d(self.hidden_size))
        l.append(nn.ReLU())

        self.encoder_net = nn.Sequential(*l)

    def forward(self, *input):
        sources, targets = input[0]

        batch_size = sources.shape[0]
        src_len = sources.shape[1]
        targ_len = targets.shape[1]
        sources_lin = self.encoder_net(sources)
        targets_lin = self.encoder_net(targets)

        sources_lin = sources_lin.view(batch_size, src_len, -1)
        targets_lin = targets_lin.view(batch_size, targ_len, -1)

        return sources_lin, targets_lin


class ReshapeLayer(nn.Module):
    """Reshape layer"""
    def __init__(self, *args):
        super(ReshapeLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def create_2_linear_batchnorm_relu_dropout(in_dim, out_dim):
    # Paper defined all sub mlps to be of depth 2
    # Relu between layers, except the final_layer
    l = []
    l.append(nn.Dropout(p=0.2))
    l.append(nn.Linear(in_dim, out_dim))
    l.append(nn.BatchNorm1d(out_dim))
    l.append(nn.ReLU())
    l.append(nn.Dropout(p=0.2))
    l.append(nn.Linear(out_dim, out_dim))
    l.append(nn.BatchNorm1d(out_dim))
    l.append(nn.ReLU())

    return nn.Sequential(*l)