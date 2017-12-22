import torch
from torch.autograd import Variable


class ReprW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, is_cuda):
        super(ReprW, self).__init__()
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._is_cuda = is_cuda

        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)


class repr_w_A_C(ReprW):
    def __init__(self, vocab_size, embedding_dim, is_cuda):
        super(repr_w_A_C, self).__init__(vocab_size, embedding_dim, is_cuda)

    def forward(self, input):
        # Tensor
        sequence = torch.unsqueeze(input,0) # batch_size = 1
        a = Variable(sequence, volatile=not self.training)
        if self._is_cuda:
            a = a.cuda()

        seq_len = a.data.shape[1]
        words_depth = a.data.shape[2]

        b = a.view(-1, seq_len * words_depth)  # Unroll to (batch, seq_len*3)
        c = self.embeddings(b)  # To (batch, seq_len*3, embed_depth)
        d = c.view(-1, seq_len, words_depth, self._embedding_dim)  # Roll to (batch, seq_len, 3, 50)
        e = d.sum(2)  # Sum along 3rd axis -> (seq_len, 50)

        return e


class repr_w_B(repr_w_A_C):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, is_cuda):
        super(repr_w_B, self).__init__(vocab_size, embedding_dim, is_cuda)
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input):
        characters = input[1]

        sentence_embeddings = []
        for word in characters:
            word_embeddings = super(repr_w_B, self).forward(word)
            sentence_embeddings.append(word_embeddings)

        word_features = []
        for word_embeddings in sentence_embeddings:
            lstm_out, __ = self.lstm(word_embeddings)
            word_features.append(lstm_out[:,-1])


        return torch.stack(word_features, dim=1)