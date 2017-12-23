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
        import batchers
        import utils

        characters = [i[1] for i in input]

        sent_length = len(characters[0])
        for sent in characters:
            assert len(sent) == sent_length # Sentences must be uniform in size

        all_words_char_seqs = [word_seq for sent_words in characters for word_seq in sent_words]  # concat lists
        org_idxs = torch.range(0, len(all_words_char_seqs)-1).long()
        batcher = batchers.Generator(all_words_char_seqs, org_idxs, flattened_labels=True)

        all_word_features = []
        all_word_idx = []
        for sub_words_list, sub_org_idx in batcher:
            sub_words_embeddings = super(repr_w_B, self).forward(sub_words_list)
            lstm_out, __ = self.lstm(sub_words_embeddings)
            sub_word_features = lstm_out[:, -1]
            all_word_features.extend(sub_word_features)
            all_word_idx.extend(sub_org_idx)

        # rearrange
        all_word_features = torch.stack(all_word_features)[all_word_idx,:]

        word_features_per_sentence = torch.split(all_word_features, sent_length, 0)

        return torch.stack(word_features_per_sentence)
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