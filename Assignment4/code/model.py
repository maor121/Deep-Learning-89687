import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear


class SNLI_Tagger(nn.Module):
    def __init__(self, embedding_size, hidden_size, labels_count, is_intra_attention=False):
        super(SNLI_Tagger, self).__init__()

        l = []

        l.append(Encode(embedding_size, hidden_size))   # encoder_net
        next_in_dim = hidden_size
        if is_intra_attention:
            l.append(IntraAttend(next_in_dim))          # 3.4 intra sentence attention_net
            next_in_dim *= 2
        l.append(Attend(next_in_dim, hidden_size))      # 3.1 attention
        l.append(Compare(2*next_in_dim, hidden_size))   # 3.2 compare
        l.append(Aggregate(2*hidden_size,hidden_size, labels_count)) # 3.3 aggregate

        self.net = nn.Sequential(*l)

    def forward(self, *input):
        return self.net(input[0])


class Aggregate(nn.Module):
    """Paper: Section 3.3"""
    def __init__(self, in_dim, hidden_size, labels_count):
        super(Aggregate, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.labels_count = labels_count

        h_mlp = create_2_linear_batchnorm_relu_dropout(in_dim, self.hidden_size)
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
    def __init__(self, in_dim, out_dim):
        super(Compare,self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = create_2_linear_batchnorm_relu_dropout(in_dim, out_dim)

    def forward(self, *input):
        alpha, beta = input[0]

        source_sent_len = alpha.shape[1]
        targets_sent_len = beta.shape[1]  # Hypothesis

        v1i = self.G(alpha.view(-1, self.in_dim)).view(-1, source_sent_len, self.out_dim)
        v2j = self.G(beta.view(-1, self.in_dim)).view(-1, targets_sent_len, self.out_dim)

        return v1i, v2j


class Attend(nn.Module):
    """Paper: Section 3.1"""
    def __init__(self, in_dim, out_dim):
        super(Attend, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.F = create_2_linear_batchnorm_relu_dropout(in_dim, out_dim)

    def _forward_F_tag(self, a, b):
        a_sent_len = a.shape[1]
        b_sent_len = b.shape[1]
        a_f_mlp = self.F(a.view(-1, self.in_dim)).view(-1, a_sent_len, self.out_dim)
        b_f_mlp = self.F(b.view(-1, self.in_dim)).view(-1, b_sent_len, self.out_dim)

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

class IntraAttend(nn.Module):
    def __init__(self, hidden_size):
        super(IntraAttend, self).__init__()

        self.hidden_size = hidden_size
        self.F_intra = create_2_linear_batchnorm_relu_dropout(hidden_size, hidden_size)

    def _forward_F_tag(self, a, b):
        a_sent_len = a.shape[1]
        b_sent_len = b.shape[1]
        a_f_mlp = self.F_intra(a.view(-1, self.hidden_size)).view(-1, a_sent_len, self.hidden_size)
        b_f_mlp = self.F_intra(b.view(-1, self.hidden_size)).view(-1, b_sent_len, self.hidden_size)

        return torch.bmm(a_f_mlp, torch.transpose(b_f_mlp, 1, 2))

    def forward(self, *input):
        sources_lin, targets_lin = input[0]

        source_sent_len = sources_lin.shape[1]
        targets_sent_len = targets_lin.shape[1]  # Hypothesis

        # Attention weights
        srcEij = self._forward_F_tag(sources_lin, sources_lin)
        trgEij = self._forward_F_tag(targets_lin, targets_lin)
        # TODO: Add di-j, dist bias, with lengh of 10. It is hard to do efficiently, a simple implementation showed no improved accuracy so I dropped it

        srcEij_softmax = F.softmax(srcEij.view(-1, source_sent_len), dim=1).view(-1, source_sent_len, source_sent_len)
        trgEij_softmax = F.softmax(trgEij.view(-1, targets_sent_len), dim=1).view(-1, targets_sent_len, targets_sent_len)

        # beta = subphrase of target which is softly aligned to source (source: paper)
        beta = torch.cat((targets_lin, torch.bmm(trgEij_softmax, targets_lin)), 2)
        # alpha = subphrase of source which is softly aligned to target (source: paper)
        alpha = torch.cat((sources_lin, torch.bmm(srcEij_softmax, sources_lin)), 2)

        return alpha, beta


class Encode(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Encode, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        l = []
        l.append(Linear(self.embedding_size, self.hidden_size))
        l.append(nn.ReLU())
        l.append(nn.Dropout(0.2))

        self.encoder_net = nn.Sequential(*l)

    def forward(self, *input):
        sources, targets = input[0]

        sources_lin = self.encoder_net(sources)
        targets_lin = self.encoder_net(targets)

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
    l.append(nn.Linear(in_dim, out_dim))
    l.append(nn.BatchNorm1d(out_dim))
    l.append(nn.ReLU())
    l.append(nn.Dropout(p=0.2))
    l.append(nn.Linear(out_dim, out_dim))
    l.append(nn.BatchNorm1d(out_dim))
    l.append(nn.ReLU())
    l.append(nn.Dropout(p=0.2))

    return nn.Sequential(*l)