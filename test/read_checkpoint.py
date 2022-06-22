import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import models.donn as donn
import pickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
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




def read_checkpoint(filename):
    """
    checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            # "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
            "mode": self.mode,
            "constrained": self.constrained,
        }
    """
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)

    train_acc_history = checkpoint["train_acc_history"]
    loss_history = checkpoint["loss_history"]
    val_acc_history = checkpoint["val_acc_history"]

    return loss_history, train_acc_history, val_acc_history
    

def main():
    loss_history, train_acc_history, val_acc_history = read_checkpoint("./temp/x0_checkpoint_epoch_40.pkl")
    plot_loss_and_accuracy(loss_history, val_acc_history, epoch=40)

if __name__ == "__main__":
    main()