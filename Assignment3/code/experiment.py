import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import TensorDataset


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, is_cuda):
        super(LSTMTagger, self).__init__()
        target_size = 2

        self.hidden_dim = hidden_dim
        self.is_cuda = is_cuda

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, target_size)

    def forward(self, sentence):
        input = Variable(sentence, volatile = not self.training)
        if self.is_cuda:
            input = input.cuda()

        input = torch.squeeze(input, 0)
        sentence_len = sentence.shape[1]

        embeds = self.word_embeddings(input)
        lstm_out, hidden = self.lstm(
            embeds.view(sentence_len, 1, -1))
        out = self.hidden2tag(lstm_out.view(sentence_len, -1))
        return torch.unsqueeze(out[-1],0)


class ModelRunner:
    def __init__(self, learning_rate, is_cuda, eval_every_n_examples):
        self.learning_rate = learning_rate
        self.is_cuda = is_cuda
        self.eval_every_n_examples = eval_every_n_examples
    def initialize_random(self, embedding_dim, hidden_dim, vocab_size):
        net = LSTMTagger(embedding_dim, hidden_dim, vocab_size, self.is_cuda)
        if (self.is_cuda):
            net.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net
    def train(self, trainloader, epoches, testloader=None, omit_tag_id=None, plot=False):
        self.net.train(True)
        if plot:
            from plot import PlotBatches
            plotter = PlotBatches()
        else:
            plotter = None
        updates_per_epoch = 0
        for epoch in range(epoches):  # loop over the dataset multiple times

            start_e_t = time.time()
            running_loss = 0.0
            examples_trained = 0
            batches_since_last_eval = 0
            for data in trainloader:
                start_b_t = time.time()

                # get the inputs
                inputs, labels = data

                # wrap labels in Variable
                labels = Variable(labels)
                if self.is_cuda:
                    labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                end_b_t = time.time()

                # print statistics
                batch_size = inputs.shape[0]
                examples_trained += batch_size
                batches_since_last_eval += 1
                running_loss += loss.data[0] * (float(batch_size) / self.eval_every_n_examples)
                if examples_trained > self.eval_every_n_examples:
                    print('[%d] loss: %.3f timer_per_batch: %.3f' %
                          (epoch + 1, running_loss, (end_b_t - start_b_t)))
                    if plot:
                        test_acc = self.eval(testloader, omit_tag_id, to_print=False)
                        plotter.update(running_loss, test_acc)
                        updates_per_epoch += 1
                        self.net.train(True)
                    running_loss = 0
                    batches_since_last_eval = 0
                    examples_trained -= self.eval_every_n_examples
            end_e_t = time.time()
            print('epoch time: %.3f' % (end_e_t - start_e_t))
        if plot:
            updates_per_epoch /= epoches
            plotter.show(updates_per_epoch, self.eval_every_n_examples)

    def eval(self, testloader, omit_tag_id=None, to_print=True):
        self.net.train(False)  # Disable dropout during eval mode
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            features, labels = data
            if self.is_cuda:
                labels = labels.cuda()
            outputs = self.net(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if omit_tag_id is not None:
                O_tag_id = omit_tag_id
                diff_O_tag = sum([1 for p, l in zip(predicted, labels) if p == l and l == O_tag_id])
                correct += (predicted == labels).sum()
                correct -= diff_O_tag
                total -= diff_O_tag
            else:
                correct += (predicted == labels).sum()
            if to_print and i % 10 == 0 and i > 0:
                print("evaluated: "+str(i))
        acc = 1.0 * correct / total
        if to_print:
            print('Accuracy of the network on the %d test words: %.3f %%' % (
                total, 100.0 * correct / total))
        return acc


def randomTrainingExample(C2I, ex_max_len):
    from gen_examples import gen_example
    example, is_positive = gen_example(ex_max_len)
    input_tensor = torch.LongTensor([C2I[c] for c in example])
    category_tensor = torch.LongTensor([is_positive])

    #return torch.unsqueeze(input_tensor, 0), torch.unsqueeze(category_tensor, 0)
    return torch.unsqueeze(input_tensor,0), category_tensor

class Generator(object):
    def __init__(self, len, C2I, ex_max_len, generateFunction):
        self._len = len
        self.current = 0
        self._C2I = C2I
        self._ex_max_len = ex_max_len
        self._gen_func = generateFunction

    def __iter__(self):
        return self

    def __len__(self):
        return self._len

    def __next__(self):
        self.current += 1
        if self.current >= self.__len__():
            raise StopIteration
        return self._gen_func(self._C2I, self._ex_max_len)

    next = __next__  # Python 2 compatibility


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


    trainloader = Generator(2500, C2I, 200, randomTrainingExample)
    testloader = Generator(200, C2I, 10000, randomTrainingExample)

    runner = ModelRunner(learning_rate, is_cuda, 50)
    runner.initialize_random(embedding_dim, hidden_dim, vocab_size)
    runner.train(trainloader, epoches)

    runner.eval(testloader)

    print('Finished Training')

