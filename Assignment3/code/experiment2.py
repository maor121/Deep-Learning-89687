import random
from experiment import Generator, LSTMTagger, ModelRunner

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
    vocab = "0123456789abcd"
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

    return input_tensor, category_tensor

if __name__ == '__main__':
    import torch.utils.data

    is_cuda = False
    embedding_dim = 50
    hidden_dim = 20
    learning_rate = 0.001
    batch_size = 1
    epoches = 1

    vocab = "0123456789abcd"
    vocab_size = len(vocab)
    C2I = {}
    for c in vocab:
        C2I[c] = len(C2I)

    res = [1 for s,is_pos in [randomTrainingExample2(C2I, 200) for i in range(1000)] if is_pos[0]==1]
    print("pos/neg: {}/{}".format(len(res),1000-len(res)))

    trainloader = Generator(2500, C2I, 200, randomTrainingExample2)
    testloader = Generator(200, C2I, 1000, randomTrainingExample2)

    runner = ModelRunner(learning_rate, is_cuda)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print('Finished Training')