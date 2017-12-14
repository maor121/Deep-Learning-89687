import random
from experiment import Generator, LSTMTagger, ModelRunner


def randomTrainingExample2(C2I, ex_max_len):
    vocab = "0123456789abcd"
    ex_len = random.randint(len(vocab) * 2 + 1, ex_max_len)

    str = ''.join(random.choice(vocab) for _ in range(ex_len))
    is_positive = str.count("a") > 2*str.count("b")

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


    trainloader = Generator(1500, C2I, 200, randomTrainingExample2)
    testloader = Generator(200, C2I, 10000, randomTrainingExample2)

    runner = ModelRunner(learning_rate, is_cuda)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print('Finished Training')