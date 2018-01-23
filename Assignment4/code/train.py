import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Sequential

from model import Encoder, Attention


def convert_batch_to_embedding(batch, w2v):
    embeddings = w2v(batch)
    return embeddings

class ModelRunner:
    def __init__(self, learning_rate, weight_decay, is_cuda):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.is_cuda = is_cuda
    def initialize_random(self, w2v, hidden_size, labels_count):
        self.w2v = w2v

        embedding_size = w2v.weight.data.shape[1]

        encoder_net = Encoder(embedding_size, hidden_size)
        attention_net = Attention(hidden_size, labels_count)
        self.net = Sequential(*[encoder_net, attention_net])
        if (self.is_cuda):
            self.net = self.net.cuda()
            self.w2v = self.w2v.cuda()

        self.criterion = nn.NLLLoss(size_average=True)
        #self.criterion = nn.CrossEntropyLoss()

        self.encoder_optimizer = torch.optim.Adagrad(encoder_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.attention_optimizer = torch.optim.Adagrad(attention_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train(self, trainloader, epoches, testloader=None):
        self.net.train(True)
        for epoch in range(epoches):  # loop over the dataset multiple times

            start_e_t = time.time()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                start_b_t = time.time()

                # get the inputs
                sources, targets, labels = data

                # Wrap tensors in variables
                sources = Variable(sources)
                targets = Variable(targets)
                labels = Variable(labels)

                if self.is_cuda:
                    sources = sources.cuda()
                    targets = targets.cuda()
                    labels = labels.cuda()

                # Convert to embeddings
                sources = convert_batch_to_embedding(sources, self.w2v)
                targets = convert_batch_to_embedding(targets, self.w2v)

                # zero the parameter gradients
                self.encoder_optimizer.zero_grad()
                self.attention_optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net((sources, targets))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.encoder_optimizer.step()
                self.attention_optimizer.step()

                end_b_t = time.time()

                # print statistics
                running_loss += loss.data[0]
                #if i % 5000 == 4999:  # print every 2000 mini-batches
                #    print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                #          (epoch + 1, i + 1, running_loss / 5000, (end_b_t - start_b_t)))
                #    running_loss = 0.0
            end_e_t = time.time()
            print('epoch time: %.3f' % (end_e_t - start_e_t))
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / i))
            if testloader is not None:
                self.eval(testloader)

    def eval(self, testloader):
        self.net.train(False)  # Disable dropout during eval mode
        correct = 0
        total = 0
        for i, data in enumerate(testloader):
            sources, targets, labels = data
            sources = Variable(sources, volatile=True)
            targets = Variable(targets, volatile=True)
            labels = Variable(labels)
            if self.is_cuda:
                labels = labels.cuda()
                sources = sources.cuda()
                targets = targets.cuda()
            sources = convert_batch_to_embedding(sources, self.w2v)
            targets = convert_batch_to_embedding(targets, self.w2v)
            outputs = self.net((sources, targets))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        print('Accuracy of the network on the %d test words: %d %%' % (
            total, 100 * correct / total))


if __name__ == '__main__':
    import utils
    train_batches, test_batches, w2v = utils.load_data(
        "../out/entail-train.hdf5","../out/entail-val.hdf5","../out/glove.hdf5"
    )

    vocab_size = w2v.weight.data.shape[0]
    lr = 0.01
    weight_decay = 5e-5
    hidden_size = 300
    epoches = 70
    labels_count = 3
    is_cuda = True

    model_runner = ModelRunner(lr, weight_decay, is_cuda)
    model_runner.initialize_random(w2v, hidden_size, labels_count)
    model_runner.train(train_batches, epoches, test_batches)

    print(0)