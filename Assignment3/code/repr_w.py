import torch
from torch.autograd import Variable


def build_padded_tensor_from_list(id_list):
    batch_size = len(id_list)
    max_seq_len = max([len(seq) for seq in id_list])
    seq_depth = id_list[0].shape[1]
    input_tensor = torch.zeros(batch_size, max_seq_len, seq_depth).long()
    lengths = []
    for i, e in enumerate(id_list):
        length = len(e)
        input_tensor[i, :length] = e
        lengths.append(length)
    return (input_tensor, lengths)


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
        sequence, lengths = build_padded_tensor_from_list(input)

        a = Variable(sequence, volatile=not self.training)
        if self._is_cuda:
            a = a.cuda()
        max_seq_len = a.data.shape[1]
        words_depth = a.data.shape[2]

        b = a.view(-1, max_seq_len * words_depth)  # Unroll to (batch, seq_len*3)
        c = self.embeddings(b)  # To (batch, seq_len*3, embed_depth)
        d = c.view(-1, max_seq_len, words_depth, self._embedding_dim)  # Roll to (batch, seq_len, 3, 50)
        e = d.sum(2)  # Sum along 3rd axis -> (batch, seq_len, 50)

        return e, lengths


class repr_w_B(ReprW):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, is_cuda):
        super(repr_w_B, self).__init__(vocab_size, embedding_dim, is_cuda)
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input):
        characters = input

        characters_embeddings = []
        for sentence in characters:
            # batch all the characters in the sentence
            sentence_padded, lengths = build_padded_tensor_from_list(sentence)
            a = Variable(sentence_padded, volatile=not self.training)
            if self._is_cuda:
                a = a.cuda()
            sentence_embeddings = self.embeddings(a)
            characters_embeddings.append((sentence_embeddings,lengths))

        word_features = []
        lengths = []
        for sentence_embeddings, lengths in characters_embeddings:
            pack = torch.nn.utils.rnn.pack_padded_sequence(sentence_embeddings, lengths, batch_first=True)
            lstm_out, __ = self.lstm(pack)
            unpack = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            last_feature_per_word = unpack[-1]

            lengths.append(len(last_feature_per_word))
            word_features.append(last_feature_per_word)

        return word_features, lengths