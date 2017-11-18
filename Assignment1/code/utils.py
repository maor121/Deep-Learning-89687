# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.

VOCABOLARY_SIZE = 600
UNK_ID = VOCABOLARY_SIZE

def read_dataset(train_set_path, dev_set_path):

    TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data(train_set_path)]
    DEV   = [(l,text_to_bigrams(t)) for l,t in read_data(dev_set_path)]

    from collections import Counter
    fc = Counter()
    for l,feats in TRAIN:
        fc.update(feats)

    # 600 most common bigrams in the training set.
    vocab = set([x for x,c in fc.most_common(VOCABOLARY_SIZE)])

    # label strings to IDs
    L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
    # feature strings (bigrams) to IDs
    F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

    return TRAIN, DEV, L2I, F2I

################################### Helper functions ####################################

def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

#Assure text is of even length, and starts and ends with _
def normalize_text(text):
    if len(text) % 2 == 0:
        text = '_' + text + '_'
    else:
        text = '_' + text + '__'
    return text

def text_to_bigrams(text):
    text = normalize_text(text)
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def dataset_to_ids(dataset, F2I, L2I):
    return [[L2I[l], [bigram_to_id(b, F2I) for b in blist]] for l,blist in iter(dataset)]

def bigram_to_id(bigram, F2I):
    return F2I[bigram] if F2I.has_key(bigram) else UNK_ID

