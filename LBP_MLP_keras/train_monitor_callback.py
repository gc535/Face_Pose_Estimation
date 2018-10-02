from keras.callbacks import Callback
from matplotlib import pyplot as plt
from IPython.display import clear_output

class Monitor(Callback):

    def on_train_begin(self, log={}):
        self.i = 0
        self.x = []
        self.accuracy = []
        self.losses = []
        self.fig = plt.figure()
        plt.ion()
        
        

    def on_epoch_end(self, epoch, log={}):
        self.x.append(self.i)
        self.losses.append(log.get('loss'))
        self.accuracy.append(log.get('acc'))
        self.i += 1

        plt.gcf().clear()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.accuracy, label="accuracy")
        plt.legend()
        plt.draw()
        plt.pause(0.001)
