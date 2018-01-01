"""Usage: experiment2.py <language_number>

-h --help    show this

"""
from docopt import docopt

import random
from experiment import Generator, ModelRunner

def check_positive(str):
    is_positive = False
    for i in range(len(str) - 5):
        all_pass = True
        for j in range(3):
            if str[i + j] != str[i + 5 - j]:
                all_pass = False
                break
        if all_pass:
            is_positive = True
            break
    return is_positive

def randomTrainingExample2(C2I, ex_max_len):
    vocab = "0123456789"
    ex_len = random.randint(len(vocab), ex_max_len)

    should_positive = random.uniform(0,1) < 0.5
    str = ''.join(random.choice(vocab) for _ in range(ex_len))
    is_positive = check_positive(str)
    if should_positive and not is_positive:
        i = random.randint(0, len(str)-5)
        sub_str = str[i:i+3]
        str = str[:i+3] + ''.join(reversed(sub_str)) + str[i+5:]
        is_positive = True
    else:
        while not should_positive and is_positive:
            str = ''.join(random.choice(vocab) for _ in range(ex_len))
            is_positive = check_positive(str)

    assert is_positive == check_positive(str)

    input_tensor = torch.LongTensor([C2I[c] for c in str])
    category_tensor = torch.LongTensor([is_positive])

    return torch.unsqueeze(input_tensor, 0), category_tensor

global prime_numbers
def randomTrainingExample3(C2I, ex_max_len):
    import numpy as np

    global prime_numbers
    if not 'prime_numbers' in globals():
        prime_numbers = []
        with open("primes.txt") as prime_file:
            for line in prime_file:
                line = line.strip()
                prime_numbers.extend([int(l) for l in line.split()])
        prime_numbers = np.array(prime_numbers)

    should_positive = random.uniform(0, 1) < 0.5
    rand_prime = random.choice(prime_numbers)
    if not should_positive:
        prime_len = len(str(rand_prime))
        r = random.randint(0, 10 ** (prime_len-1))
        t = random.choice([1,3,5,7,9])
        rand_prime = r * 10 + t
    rand_prime = str(rand_prime)

    input_tensor = torch.LongTensor([C2I[c] for c in rand_prime])
    category_tensor = torch.LongTensor([should_positive])

    return torch.unsqueeze(input_tensor, 0), category_tensor


def randomTrainingExample4(C2I, ex_max_len):

    ex_len = random.randint(1, ex_max_len)
    should_positive = random.uniform(0, 1) < 0.5

    if should_positive:
        r = random.randint(0, ex_len/3) * 7
    else:
        r = random.randint(0,ex_len)
        while r % 7 == 0:
            r = random.randint(0,ex_len)
    str_r = str(r)

    input_tensor = torch.LongTensor([C2I[c] for c in str_r])
    category_tensor = torch.LongTensor([should_positive])

    return torch.unsqueeze(input_tensor, 0), category_tensor



if __name__ == '__main__':
    import torch.utils.data

    arguments = docopt(__doc__, version='Naval Fate 2.0')
    language_number = arguments['<language_number>']

    if (not language_number.isdigit()) or language_number < 0 or language_number > 3:
        print("language_number must be a number between 1-3")
    language_number = int(language_number)

    is_cuda = False
    embedding_dim = 50
    hidden_dim = 20
    learning_rate = 0.001
    epoches = 1

    vocab = "0123456789"
    vocab_size = len(vocab)
    C2I = {}
    for c in vocab:
        C2I[c] = len(C2I)

    if language_number == 1:
        train_num = 2500
        test_num = 200
        train_max_len = 200
        test_max_len = 1000
        func = randomTrainingExample2
    else:
        if language_number == 2:
            train_num = 7500
            test_num = 300
            train_max_len = test_max_len = 1
            func = randomTrainingExample3
        else:
            #if language_number == 3:
            train_num = 7500
            test_num = 300
            train_max_len = 2500
            test_max_len = 1000
            func = randomTrainingExample4

    #res = [1 for s,is_pos in [randomTrainingExample2(C2I, 200) for i in range(1000)] if is_pos[0]==1]
    #print("pos/neg: {}/{}".format(len(res),1000-len(res)))

    trainloader = Generator(train_num, C2I, train_max_len, func)
    testloader = Generator(test_num, C2I, test_max_len, func)

    runner = ModelRunner(learning_rate, is_cuda, 50)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print('Finished Training')