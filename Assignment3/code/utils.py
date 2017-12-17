import re
import torch
import torch.nn as nn
import torch.nn.functional as F

RARE_WORDS_MAX_COUNT = 3
RARE_FEATURE_MAX_COUNT = 3

FEATURES_PER_WORD = 2

DIGIT_PATTERN = re.compile('\d')


def parse_word_tag(line, is_tagged, replace_numbers, lower_case):
    tag = None
    if is_tagged:
        try:
            word, tag = line.split()
            if replace_numbers:
                word = re.sub(DIGIT_PATTERN, '#', word)
            if lower_case:
                word = word.lower()
        except Exception:
            is_tagged = False
            word = line
    else:
        word = line
    return is_tagged, word, tag


def scan_input_file(path, W2I=None, T2I=None, F2I=None,
                    UNK_WORD="*UNK*", START_WORD="*START*", END_WORD="*END*",
                    lower_case=False, replace_numbers=True, calc_sub_word=False):
    calc_W = W2I == None
    calc_T = T2I == None
    calc_F = F2I == None and calc_sub_word
    if calc_W:
        W2I = StringCounter([START_WORD, END_WORD, UNK_WORD], UNK_WORD)
    else:
        # Pretrained (probably not necessary)
        W2I.get_id_and_update(START_WORD)
        W2I.get_id_and_update(END_WORD)
    if calc_T:
        T2I = StringCounter()
    if calc_F:
        F2I = StringCounter([UNK_WORD], UNK_WORD=UNK_WORD)

    num_words = 0
    saw_empty_line = True
    is_tagged = True
    with open(path) as data_file:
        for line in data_file:
            line = line.strip()
            if len(line) > 0: # Not end of sentence
                is_tagged, w, t = parse_word_tag(line, is_tagged, replace_numbers, lower_case)
                if calc_W:
                    W2I.get_id_and_update(w)
                if calc_T and is_tagged:
                    T2I.get_id_and_update(t)
                num_words += 1
                saw_empty_line = False
            else:
                if not saw_empty_line:
                    if calc_W:
                        # Count appearences of START, END
                        W2I.get_id_and_update(START_WORD)
                        W2I.get_id_and_update(END_WORD)
                    if calc_T:
                        T2I.get_id_and_update(START_WORD)
                        T2I.get_id_and_update(END_WORD)
                    num_words += 2
                saw_empty_line = True

    # Calc word features
    if calc_F:
        for word, count in W2I.S2C.iteritems():
            extract_features(word, F2I, updateF2I=True, count=count)
        # Filter rare features
        F2I.filter_rare_words(RARE_FEATURE_MAX_COUNT + 1)
        F2I = StringCounter(F2I.S2I.keys(), F2I.UNK_WORD)

    # Filter rare words
    if calc_W:
        W2I.filter_rare_words(RARE_WORDS_MAX_COUNT+1)
        W2I = StringCounter(W2I.S2I.keys(), W2I.UNK_WORD)
        assert START_WORD in W2I.S2I
        assert END_WORD in W2I.S2I
        assert UNK_WORD in W2I.S2I

    if calc_F:
        F2I.shift_ids_by(W2I.len())

    return num_words, W2I, T2I, F2I

def extract_features(word, F2I, updateF2I, count=1):
    """Return a torch.LongTensor of features per word."""

    prefix_3 = word[:3]  # Will work even for words of size < 3
    suffix_3 = word[-3:]
    if updateF2I:
        prefix_3_id = F2I.get_id_and_update(prefix_3)
        suffix_3_id = F2I.get_id_and_update(suffix_3)
    else:
        prefix_3_id = F2I.get_id(prefix_3)
        suffix_3_id = F2I.get_id(suffix_3)
    return [prefix_3_id, suffix_3_id]


def load_dataset(path, W2I=None, T2I=None, F2I=None,
                 UNK_WORD="*UNK*", START_WORD="*START*", END_WORD="*END*",
                 lower_case=False, replace_numbers=True, calc_sub_word=False):
    num_words, W2I, T2I, F2I = scan_input_file(path, W2I=W2I, T2I=T2I, F2I=F2I,
                                    UNK_WORD=UNK_WORD, START_WORD=START_WORD, END_WORD=END_WORD,
                                    lower_case=lower_case, replace_numbers=replace_numbers,
                                    calc_sub_word=calc_sub_word)

    train_w_depth = FEATURES_PER_WORD+1 if calc_sub_word else 1
    input_tensor = torch.LongTensor(num_words, train_w_depth)
    labels_tensor = torch.LongTensor(num_words)

    sentence = []
    sentence_labels = []
    word_index = 0
    saw_empty_line = True
    is_tagged = True
    with open(path) as data_file:
        for line in data_file:
            line = line.strip()
            if len(line) > 0:
                is_tagged, w, t = parse_word_tag(line, is_tagged, replace_numbers, lower_case)
                sentence.append(w)
                if is_tagged:
                    sentence_labels.append(T2I.get_id(t))
                word_index += 1
                saw_empty_line = False
            else:
                if not saw_empty_line: # END of sentence
                    sentence = [START_WORD] + sentence + [END_WORD]
                    sentence_len = len(sentence)
                    sentence_ids = [W2I.get_id(w) for w in sentence]
                    word_index += 2

                    input_tensor[word_index-sentence_len:word_index,0] = torch.LongTensor(sentence_ids)
                    if is_tagged:
                        sentence_labels = [T2I.get_id(START_WORD)] + sentence_labels + [T2I.get_id(END_WORD)]
                        labels_tensor[word_index-sentence_len:word_index] = torch.LongTensor(sentence_labels)

                    if F2I is not None:
                        features_ids = [extract_features(w, F2I, updateF2I=False) for w in sentence]

                        input_tensor[word_index-sentence_len:word_index,1:] = torch.LongTensor(features_ids)
                saw_empty_line = True
                sentence = []
                sentence_labels = []

    if not is_tagged:
        print "blind file detected!"
        labels_tensor[:] = 0 # No real value

    if calc_sub_word:
        return W2I, T2I, F2I, input_tensor, labels_tensor
    else:
        return W2I, T2I, input_tensor, labels_tensor


def list_to_tuples(L, tup_size):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3,s4), ..."
    from itertools import tee, izip
    tupItr = tee(L, tup_size)
    for i, itr in enumerate(tupItr):
        for j in range(i):
            next(itr, None)
    return izip(*tupItr)


def inverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}


class StringCounter:
    def __init__(self, initialStrList=[], UNK_WORD=None):
        from collections import Counter
        self.S2I = {}
        self.S2C = Counter()
        self.last_id = 0
        self.UNK_WORD = UNK_WORD
        for s in initialStrList:
            self.get_id_and_update(s)

    def get_id_and_update(self, str, count=1):
        if not self.S2I.__contains__(str):
            self.S2I[str] = self.last_id
            self.last_id += 1
        self.S2C[str] = self.S2C.get(str, 0) + count
        return self.S2I[str]

    def get_id(self, str):
        if not self.S2I.__contains__(str):
            str = self.UNK_WORD
        return self.S2I[str]

    def filter_rare_words(self, min_count):
        w_to_filter = [k for k, v in self.S2C.iteritems() if v < min_count]
        for w in w_to_filter:
            self.S2C.pop(w)
            self.S2I.pop(w)
        self.get_id_and_update(self.UNK_WORD)

    def len(self):
        return len(self.S2I)

    def shift_ids_by(self, n):
        S2I = {}
        for k, v in self.S2I.iteritems():
            S2I[k] = v + n
        self.S2I = S2I

