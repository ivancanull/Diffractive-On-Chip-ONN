import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import models.donn as donn
import models.flexible_donn as flexible_donn

import pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

def plot_loss_and_accuracy(loss, accuracy, epoch):

    X_loss = np.linspace(0, epoch, 480)
    selelct_loss = []
    for i in range(480):
        selelct_loss.append(loss[i * 100])
    X_accuracy = np.linspace(0, epoch, len(accuracy))

    color = 'blue'
    fig, ax_loss = plt.subplots()
    
    ax_loss.set_xlabel('epoch',)

    ax_loss.plot(X_loss, selelct_loss, color=color)
    ax_loss.set_ylabel('loss', color=color)
    ax_loss.tick_params(axis='y', color=color)

    color='red'
    ax_accuracy = ax_loss.twinx()
    ax_accuracy.plot(X_accuracy, accuracy, color=color)
    ax_accuracy.set_ylabel('accuracy', color=color)
    ax_accuracy.tick_params(axis='y', color=color)

    fig.savefig(os.path.join('./figures', 'loss_accuracy.pdf'))
    fig.show()

class checkpoint(object):

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint["model"]
        self.update_rule = checkpoint["update_rule"]
        self.lr_decay = checkpoint["lr_decay"]
        self.batch_size = checkpoint["batch_size"]
        self.num_train_samples = checkpoint["num_train_samples"]
        self.num_val_samples = checkpoint["num_val_samples"]
        self.epoch = checkpoint["epoch"]
        self.train_acc_history = checkpoint["train_acc_history"]
        self.val_acc_history = checkpoint["val_acc_history"]
        self.mode = checkpoint["mode"]
        self.constrained = checkpoint["constrained"]
    
    def get_model(self):
        return self.model
    
    def write_x0_diff(self):
        for layer_index, layer in enumerate(self.model.layers):
            diff = (layer.x0 - layer.original_x0) / 1e-6
            np.savetxt("./temp/diff_layer_" + str(layer_index) + ".txt", diff)
        return

def main(args):

    filename = "./temp/" + args.filename + ".pkl"
    cp = checkpoint(filename)
    cp.write_x0_diff()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', dest='filename', default='temp_epoch_10', type=str)
    args = parser.parse_args()

    main(args)