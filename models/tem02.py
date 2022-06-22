import numpy as np
import pandas as pd
import torch
import os
from os import path
from matplotlib import pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
    def forward(self, x):
        x = torch.tanh(self.hidden(x))      # activation function for hidden layer 
        x = self.predict(x)             # linear output
        return x

class TEM02:
    def __init__(self, 
                 input_num=1, 
                 neuron_num=20, 
                 output_num=2, 
                 iter_num=10000,
                 retrain=False):

        self.cur_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.cur_path, '../../../Data/TEM02.csv')
        self.data = np.array(pd.read_csv(self.data_path))
        
        self.net = Net(n_feature=input_num, n_hidden=neuron_num, n_output=output_num)
        self.pt_path = os.path.join(self.cur_path, './temp/TEM02.pt')
        self.npz_path = os.path.join(self.cur_path, './temp/TEM02.npz')
        if path.exists(self.pt_path) and path.exists(self.npz_path) and retrain == False:
            self.net.load_state_dict(torch.load(self.pt_path))
            loaded = np.load(self.npz_path)
            self.mean = loaded['mean']
            self.max = loaded['max']
            self.std = loaded['std']
            self.data = (self.data - self.mean) / self.std
            self.x = torch.tensor(self.data[:, 0:input_num]).float()
            self.y = torch.tensor(self.data[:, input_num:input_num + output_num]).float()
        else:
            self.mean = self.data.mean(axis=0)
            self.max = self.data.max(axis=0)
            self.std = self.data.std(axis=0)
            np.savez(self.npz_path, mean=self.mean, max=self.max, std=self.std)
            self.data = (self.data - self.mean) / self.std
            self.x = torch.tensor(self.data[:, 0:input_num]).float()
            self.y = torch.tensor(self.data[:, input_num:input_num + output_num]).float()
            self.iter_num = iter_num
            self.train()
        
    def train(self):
              
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        loss = []
        for i in range(self.iter_num):
            prediction = self.net(self.x)    # input x and predict based on x
            loss = loss_func(prediction, self.y)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
        torch.save(self.net.state_dict(), self.pt_path)
    
    def inference(self, X):
        """
        Give the Ex and Ey Results Inside the Neurons
        Args:
            X: x_coords of arbitrary shape (m, n, l)
        Returns: 
            Ex and Ey Results of shape (m, n, l, 2)
        """

        original_shape = X.shape

        # convert the original shape (m, n, l) to (m * n * l, 1)
        X = X.reshape((-1, 1))

        prediction = self.net(torch.tensor((X - self.mean[0]) / self.std[0]).float())
        prediction = prediction.detach().numpy()
        prediction = prediction * self.std[1:3]
        prediction = prediction + self.mean[1:3]

        prediction = prediction.reshape(original_shape + (2,))
        return prediction

    def verify(self):
        
        real_x = self.x * self.std[0] + self.mean[0]
        prediction = self.net(self.x)

        real_Ex = self.y.numpy()[:,0] * self.std[1] + self.mean[1]
        pred_Ex = prediction.detach().numpy()[:,0] * self.std[1] + self.mean[1]
        real_Ey = self.y.numpy()[:,1] * self.std[2] + self.mean[2]
        pred_Ey = prediction.detach().numpy()[:,1] * self.std[2] + self.mean[2]

        Ex_SST = np.sum((real_Ex - np.mean(real_Ex)) ** 2)
        Ex_SSE = np.sum((pred_Ex - real_Ex) ** 2)
        Ex_R2 = 1 - Ex_SSE / Ex_SST
        print("Ex R2: ", Ex_R2)
        
        Ey_SST = np.sum((real_Ey - np.mean(real_Ey)) ** 2)
        Ey_SSE = np.sum((pred_Ey - real_Ey) ** 2)
        Ey_R2 = 1 - Ey_SSE / Ey_SST
        print("Ey R2: ", Ey_R2)
        
        
        plt.figure(figsize=(12,8))
        plt.subplot(1,2,1)
        plt.scatter(real_x, pred_Ex, label='pred')
        plt.scatter(real_x, real_Ex, label='real')
        plt.legend()
        plt.xlabel("position")
        plt.ylabel("Ex")
        plt.title('TEM02 Ex')  
        
        plt.subplot(1,2,2)
        plt.scatter(real_x, pred_Ey, label='pred')
        plt.scatter(real_x, real_Ey, label='real')
        plt.legend()
        plt.xlabel("position")
        plt.ylabel("Ey")
        plt.title('TEM02 Ey')  
        
        plt.show()

def main():
    test_TEM02 = TEM02()
    test_TEM02.verify()

    
if __name__ == '__main__':
    main()