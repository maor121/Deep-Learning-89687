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
        sequence = torch.stack(input,0) # first dim is batch. all of the same length
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
    def out_dim(self):
        return self._embedding_dim

class repr_w_B(repr_w_A_C):
    def __init__(self, num_chars, embedding_dim, hidden_dim, is_cuda):
        super(repr_w_B, self).__init__(num_chars, embedding_dim, is_cuda)
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input):
        characters = [i[1] for i in input]

        batch_word_features = []
        for sentence_characters in characters:
            sentence_embeddings = []
            for word in sentence_characters:
                word_embeddings = super(repr_w_B, self).forward([word])
                sentence_embeddings.append(word_embeddings)

            word_features = []
            for word_embeddings in sentence_embeddings:
                lstm_out, __ = self.lstm(word_embeddings)
                word_features.append(lstm_out[:,-1])

            batch_word_features.append(torch.stack(word_features, dim=1))

        return torch.cat(batch_word_features, 0)
    def out_dim(self):
        return self.hidden_dim

class repr_w_D(torch.nn.Module):
    def __init__(self, num_words, num_characters, char_embed_dim, char_dim_out, word_embed_dim, hidden_dim, is_cuda):
        super(repr_w_D, self).__init__()
        self.repr_w_words = repr_w_A_C(num_words, word_embed_dim, is_cuda)
        self.repr_w_chars = repr_w_B(num_characters, char_embed_dim, char_dim_out, is_cuda)

        linear_in_dim = self.repr_w_words.out_dim() + self.repr_w_chars.out_dim()
        self.linear_out = hidden_dim
        self.linear_layer = torch.nn.Linear(linear_in_dim, self.linear_out)

        if is_cuda:
            self.cuda()

    def forward(self, input):
        words = [i[0] for i in input]
        word_emb = self.repr_w_words(words)
        char_emb = self.repr_w_chars(input)

        linear_in = torch.cat([word_emb, char_emb], 2)

        return self.linear_layer(linear_in)
    def out_dim(self):
        return self.linear_out