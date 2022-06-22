import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

from utils import optim
from utils import constants as Const
import encoding.utils


class MLPModel(torch.nn.Module):

    def __init__(self,
                input_neuron_num,
                hidden_neuron_num,
                output_neuron_num,
                hidden_layer_num,):
        super(MLPModel, self).__init__()

        self.input_neuron_num = input_neuron_num
        self.hidden_neuron_num = hidden_neuron_num
        self.output_neuron_num = output_neuron_num
        self.input_layer = torch.nn.Linear(input_neuron_num, hidden_neuron_num)
        self.hidden_layer = torch.nn.Linear(hidden_neuron_num, hidden_neuron_num)
        self.output_layer = torch.nn.Linear(hidden_neuron_num, output_neuron_num)
        self.activation_layer = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.hidden_layer_num = hidden_layer_num
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_layer(x)
        for i in range(self.hidden_layer_num - 1):
            x = self.hidden_layer(x)
            x = self.activation_layer(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        torch.nn.init.constant_(m.bias, 0.0)

class MLP_Solver(object):

    def __init__(self, 
                 model, 
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

        # define the modulated mode
        # self.mode = kwargs.pop("mode", "phi")
        # self.constrained = kwargs.pop("constrained", True)

        self.config = {}
        self.config["learning_rate"] = kwargs.pop("learning_rate", 1e-14)

        self.print_every = kwargs.pop("print_every", 10)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], momentum=0.9)
        self.loss = torch.nn.MSELoss()

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

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
    
        X_batch = torch.Tensor(self.X_train[batch_mask])
        y_batch = torch.zeros((self.batch_size, self.model.output_neuron_num))

        indices = torch.unsqueeze(torch.LongTensor(self.y_train[batch_mask]), 1)
        # indices = torch.LongTensor(self.y_train[batch_mask])
        y_batch = y_batch.scatter(1, indices, 1)
        self.optimizer.zero_grad()
        prediction = self.model(X_batch)
        output = self.loss(prediction, y_batch)
        output.backward()
        self.optimizer.step()

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
            X = torch.Tensor(X[mask])
            y = y[mask]

        # compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            inference = self.model(X[start:end])
            # print(torch.argmax(inference, axis=1).detach().numpy())
            y_pred.append(torch.argmax(inference, axis=1).detach().numpy()) 
        y_pred = np.hstack(y_pred)
        # print(y_pred == y)
        acc = np.mean(y_pred == y)
        return acc

    def train(self):

        self.model.apply(weights_init)
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


                print(
                    "(Epoch %d / %d) train acc: %f; val_acc: %f"
                    % (self.epoch, self.num_epochs, train_acc, val_acc,)
                )
                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model = self.model

        # At the end of training swap the best params into the model
        return self.best_model

        


def test_MLP():
    new_size = 14
    input_dim = new_size ** 2
    mlpmodel = MLPModel(input_neuron_num=input_dim,
                        hidden_neuron_num=400,
                        output_neuron_num=10,
                        hidden_layer_num=2)

    (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(new_size)
    data = {}
    data["X_train"] = train_X / 255.0
    data["y_train"] = train_y
    data["X_val"] = test_X / 255.0
    data["y_val"] = test_y
    
    print('The model:')
    print(mlpmodel)
    solver = MLP_Solver(mlpmodel, data,
                        learning_rate=1e-3,
                        num_epochs=1000,
                        batch_size=50,
                        verbose=False,
                        lr_decay=0.95,
                        checkpoint_name="x0_checkpoint",
                        )
            
    solver.train()
    # print('\n\nModel params:')
    # for param in mlpmodel.parameters():
    #     print(param)

    # print('\n\nLayer params:')
    # for param in mlpmodel.hidden_layer.parameters():
    #     print(param)



def main():
    test_MLP()

if __name__ == '__main__':
    main()
    pass
