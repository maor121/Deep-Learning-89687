import time
import torch
from torch import nn
from torch.autograd import Variable
from model import SNLI_Tagger

def convert_batch_to_embedding(batch, w2v):
    # TODO
    pass

class ModelRunner:
    def __init__(self, learning_rate, is_cuda):
        self.learning_rate = learning_rate
        self.is_cuda = is_cuda
    def initialize_random(self, w2v):
        net = SNLI_Tagger()
        if (self.is_cuda):
            net.cuda()

        self.w2v = w2v

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        self.net = net
    def train(self, trainloader, epoches):
        self.net.train(True)
        for epoch in range(epoches):  # loop over the dataset multiple times

            start_e_t = time.time()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                start_b_t = time.time()

                # get the inputs
                sources, targets, labels = data
                sources = convert_batch_to_embedding(sources, self.w2v)
                targets = convert_batch_to_embedding(targets, self.w2v)

                # Wrap tensors in variables
                sources = Variable(sources)
                targets = Variable(targets)
                labels = Variable(labels)

                if self.is_cuda:
                    sources = sources.cuda()
                    targets = targets.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(sources, targets)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                end_b_t = time.time()

                # print statistics
                running_loss += loss.data[0]
                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f timer_per_batch: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50, (end_b_t - start_b_t)))
                    running_loss = 0.0
            end_e_t = time.time()
            print('epoch time: %.3f' % (end_e_t - start_e_t))

    def eval(self, testloader):
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
            correct += (predicted == labels).sum()
            if i % 10 == 0 and i > 0:
                print("evaluated: "+str(i))
        print('Accuracy of the network on the %d test words: %d %%' % (
            total, 100 * correct / total))


if __name__ == '__main__':
    import utils
    train_batches, test_batches, w2v = utils.load_data(
        "../out/entail-train.hdf5","../out/entail-val.hdf5","../out/glove.hdf5"
    )

    vocab_size = len(w2v)
    lr = 0.001
    is_cuda = True

    model_runner = ModelRunner(lr, is_cuda)
    model_runner.initialize_random()


    print(0)