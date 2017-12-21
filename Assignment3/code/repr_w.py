import torch
from torch.autograd import Variable

import utils


def build_padded_variable_from_list(id_list):
    max_seq_len = max([len(seq) for seq in id_list])
    input_tensors = []
    lengths = []
    for i, e in enumerate(id_list):
        length = len(e)
        input_tensors.append(torch.nn.functional.pad(e, (0,0,0,max_seq_len-length)))
        lengths.append(length)
    input_tensor = torch.stack(input_tensors)
    return (input_tensor, lengths)


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

        if self._is_cuda:
            self.cuda()

class repr_w_A_C(ReprW):
    def __init__(self, vocab_size, embedding_dim, is_cuda):
        super(repr_w_A_C, self).__init__(vocab_size, embedding_dim, is_cuda)

    def forward(self, input):
        # Tensor
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


class repr_w_B(repr_w_A_C):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, is_cuda):
        super(repr_w_B, self).__init__(vocab_size, embedding_dim, is_cuda)
        self.hidden_dim = hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input):
        characters = [i[1] for i in input]

        characters_embeddings = []
        for sentence in characters:
            # batch all the characters in the sentence
            sen_order = range(len(sentence))
            sentence, sen_order = utils.sort_by_len(sentence, sen_order)
            sentence_embeddings, lengths = super(repr_w_B, self).forward(sentence)
            characters_embeddings.append((sentence_embeddings,lengths, sen_order))

        word_features = []
        lengths = []
        for sentence_embeddings, sen_lengths, sen_order in characters_embeddings:
            #[words_in_sentence,max_word_len,embed_depth]
            pack = torch.nn.utils.rnn.pack_padded_sequence(sentence_embeddings, sen_lengths, batch_first=True)
            lstm_out, __ = self.lstm(pack)
            unpack, __ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            last_feature_per_word = [unpack[i,l-1,:] for i, l in enumerate(sen_lengths)] #last feature per word
            last_feature_per_word = torch.stack(last_feature_per_word)

            last_feature_per_word = last_feature_per_word[sen_order,:]

            lengths.append(len(last_feature_per_word))
            word_features.append(last_feature_per_word)
        # rearrange

        word_features, __ = build_padded_variable_from_list(word_features)
        return word_features, lengths