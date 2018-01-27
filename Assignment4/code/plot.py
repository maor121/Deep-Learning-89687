from matplotlib import pyplot as plt
import numpy as np


class PlotBatches(object):
    def __init__(self):
        self.train_history = {"train_loss" : [], "test_loss" : [], "test_acc": []}

    def update(self, train_loss, test_loss, test_acc):
        self.train_history['train_loss'].append(train_loss)
        self.train_history['test_loss'].append(test_loss)
        self.train_history['test_acc'].append(test_acc)
    def show(self, updates_per_epoch):
        train_loss = np.array([i for i in self.train_history['train_loss']])
        test_loss = np.array([i for i in self.train_history['test_loss']])
        test_acc = np.array([i for i in self.train_history['test_acc']])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.gca().cla()
        plt.xlabel("epoches")
        plt.plot(train_loss, label="Train Loss")
        plt.plot(test_loss, label="Test Loss")
        plt.plot(test_acc, label="Test Accuracy")


        # scale
        #ticks = plt.gca().get_xticks() * (1.0/updates_per_epoch)
        #ticks = np.round(ticks, 2)
        #plt.gca().set_xticklabels(ticks)

        # Annotate epoches points
        last_reported_epoch = None
        for i, y_val in enumerate(test_acc):
            epoch = (i+1) / updates_per_epoch
            if epoch != last_reported_epoch and epoch > 0:
                ax.annotate("%.3f" % test_acc[i], xy=(epoch*updates_per_epoch,test_acc[i]), textcoords='data')
                last_reported_epoch = epoch

        #plt.legend()
        #plt.draw()
        plt.grid()
        plt.show()

import re
EXTRACT_FLOAT_LIST_REGEX = re.compile(r"[-+]?\d*\.\d+|\d+")
def extract_epoch_info(line1, line2, line3):
    epoch_num = int(line1.split(',')[0][1:])

    train_loss, train_acc = tuple(re.findall(EXTRACT_FLOAT_LIST_REGEX, line2))
    dev_loss, dev_acc = tuple(re.findall(EXTRACT_FLOAT_LIST_REGEX, line3))

    return (epoch_num, float(train_loss), float(train_acc), float(dev_loss), float(dev_acc))

def plot(epoches_info):
    graph = PlotBatches()
    for epoch in epoches_info:
        epoch_num, train_loss, train_acc, dev_loss, dev_acc = epoch
        graph.update(train_loss, dev_loss, dev_acc)

    graph.show(len(epoch_info))


if __name__ == '__main__':
    import sys
    log_file = sys.argv[1]

    with open(log_file) as log_file:
        log_iter = iter(log_file)
        line1 = log_iter.next()
        while not line1.__contains__("epoch time"):
            line1 = log_iter.next()

        epoches_info = []
        while True:
            try:
                line2 = log_iter.next()
                line3 = log_iter.next()

                epoch_info = extract_epoch_info(line1,line2,line3)
                epoches_info.append(epoch_info)

                line1 = log_iter.next()
            except StopIteration:
                break

        #print(epoches_info)

        plot(epoches_info)
