"""Usage: gen_examples.py [-p number] [-n number] [-l number] [OUT_POSITIVE_FILE] [OUT_NEGATIVE_FILE]

-h --help    show this
-p number    number of positive examples to generate [default: 500]
-n number    number of negative examples to generate [default: 500]
-l number    max length per example [default: 1000]

"""
from docopt import docopt
import random
import string

def gen_digit_pad(len):
    return ''.join(random.choice(string.digits) for _ in range(len))

def gen_letter_pad(ch, pad_len):
    return ch*pad_len

def generate_string(is_positive, limit):
    if is_positive:
        letters = ['a', 'b', 'c', 'd']
    else:
        letters = ['a', 'c', 'b', 'd']
    #randomize example len
    ex_len = random.randint(len(letters)*2+1, limit)

    #digits pad sections are len(letters) *2+1
    pad_ids = sorted(random.sample(xrange(1, ex_len), len(letters)*2-1)) # random ids in range without repeat
    pad_ids = pad_ids + [ex_len]

    sub_strs = []
    last_index = 0
    letterIndex = 0
    is_digit_pad = True
    for p_id in pad_ids:
        sub_str_len = p_id-last_index
        if is_digit_pad:
            sub_strs.append(gen_digit_pad(sub_str_len))
        else:
            sub_strs.append(gen_letter_pad(letters[letterIndex], sub_str_len))
            letterIndex += 1
        is_digit_pad = not is_digit_pad
        last_index = p_id

    return ''.join(sub_strs)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    out_positive_filename = arguments['OUT_POSITIVE_FILE']
    out_negative_filename = arguments['OUT_NEGATIVE_FILE']
    p_count = int(arguments['-p'])
    n_count = int(arguments['-n'])
    ex_max_len = int(arguments['-l'])

    with open(out_positive_filename, "w+") as pos_file,\
        open(out_negative_filename, "w+") as neg_file:
        for i in range(p_count):
            pos_file.write(generate_string(True, ex_max_len)+"\n")
        for i in range(n_count):
            neg_file.write(generate_string(True, ex_max_len) + "\n")

    print("Done")