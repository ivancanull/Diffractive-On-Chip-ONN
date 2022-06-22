import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

from utils import optim
from utils import constants as Const
import numpy as np
import models.donn as donn
import pickle as pickle

class Solver(object):

    def __init__(self, 
                 model: donn.DONN, 
                 data, 
                 **kwargs):
        """
        
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.batch_size = kwargs.pop("batch_size", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", 1000)
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.checkpoint_name = kwargs.pop("checkpoint_name", None)

        # define the modulated mode
        self.mode = kwargs.pop("mode", "phi")
        self.constrained = kwargs.pop("constrained", True)

        self.config = {}
        self.config["learning_rate"] = kwargs.pop("learning_rate", 1e-14)

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        self.print_every = kwargs.pop("print_every", 10)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_model = None
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
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
        filename = "./temp/%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def _test(self):

        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        inference, loss, dout_r_list, dout_i_list, dh_list = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for layer_index, layer in enumerate(self.model.layers):
            w = layer.h_neuron
            dw = dh_list[layer_index]
            next_w, next_config = self.update_rule(w, dw, self.config)
            self.config = next_config
            layer.h_neuron = next_w
            #print("dw: ", dw)
            #print("dout_r_list :", dout_r_list[layer_index])

            # set the threshold value
            layer.h_neuron[layer.h_neuron < 1 * Const.Lambda0] = 1 * Const.Lambda0
            layer.h_neuron[layer.h_neuron > 3 * Const.Lambda0] = 3 * Const.Lambda0

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        if self.mode == "phi":
            output_Ex, loss, dw_list = self.model.loss_v3(X_batch, y=y_batch)
        elif self.mode == "x0":
            output_Ex, loss, dw_list = self.model.loss_v4(X_batch, y=y_batch)
        
        # here loss is a [batch_size, 1] array
        self.loss_history.append(loss)

        for layer_index, layer in enumerate(self.model.layers):
            if self.mode == "phi":
                w = layer.phi
            elif self.mode == "x0":
                w = layer.x0
            dw = dw_list[layer_index]
            next_w, next_config = self.update_rule(w, dw, self.config)
            self.config = next_config
            #print(dw)
            # set the threshold value
            if self.mode == "phi":
                layer.phi = next_w
                layer.phi[layer.phi < 0] = 0
                layer.phi[layer.phi >= 2 * np.pi] = 2 * np.pi
            elif self.mode == "x0":
                #print((next_w - w) / Const.Lambda0)
                layer.x0 = next_w
                if self.constrained == True:
                    x0_outbound_l = layer.x0 < layer.x0_left_limit
                    x0_outbound_r = layer.x0 > layer.x0_right_limit
                    layer.x0[x0_outbound_l] = layer.x0_left_limit[x0_outbound_l]
                    layer.x0[x0_outbound_r] = layer.x0_right_limit[x0_outbound_r]
                
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            output_Ex = self.model.forward(X[start:end])
            inference = np.argmax(np.abs(output_Ex), axis=1)
            y_pred.append(inference)
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc
    
    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, np.mean(self.loss_history[-1]))
                )
                #print(self.model.layers[0].h_neuron)
                #self._test()

            # at the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                self.config["learning_rate"] = self.config["learning_rate"] * self.lr_decay

            # check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = t == 0
            last_it = t == num_iterations - 1

            
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()
                
                for layer_index, layer in enumerate(self.model.layers):
                    if self.verbose and self.mode == "x0":
                        print ("delta x0 for layer %d:" % layer_index, (layer.x0 - layer.original_x0) / Const.Lambda0)

                print(
                    "(Epoch %d / %d) train acc: %f; val_acc: %f"
                    % (self.epoch, self.num_epochs, train_acc, val_acc)
                )

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model = self.model

        # At the end of training swap the best params into the model
        return self.best_model
