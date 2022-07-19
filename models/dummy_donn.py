from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
from tabnanny import verbose
from matplotlib import pyplot
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import cv2

from utils import constants as Const
import utils.helpers
#import utils.plot as plot
import models.onn_layer
import models.layers as Layers
import encoding.utils
from models.flexible_donn import Flexible_DONN as Flexible_DONN

class Dummy_DONN(object):

    def __init__(self,
                input_neuron_num,
                input_distance,
                input_bound,
                output_neuron_num,
                output_distance,
                output_bound,
                hidden_layer_num,
                hidden_neuron_num,
                hidden_distance,
                hidden_bound,
                hidden_layer_distance,
                output_layer_distance,
                phi_init="default",
                nonlinear=False,
                ) -> None:
    
        self.layers = []    
        self.dests = []
        self.Ex_out = []
        self.Ex_cache = []

        self.dw = []
        
        self.hidden_layer_num = hidden_layer_num
        self.layer_num = hidden_layer_num + 1

        self.input_distance = input_distance
        self.input_bound = input_bound

        self.hidden_distance = hidden_distance
        self.hidden_bound = hidden_bound

        self.output_distance = output_distance
        self.output_bound = output_bound

        self.input_neuron_num = input_neuron_num
        self.hidden_neuron_num = hidden_neuron_num
        self.output_neuron_num = output_neuron_num

        input_layer = models.onn_layer.ONN_Layer(neuron_number=input_neuron_num, 
                                                 distance=input_distance, 
                                                 bound=input_bound,
                                                 y=0,
                                                 phi_init=phi_init,
                                                 )
        
        self.layers.append(input_layer)
        
        # dummy donn hidden mask
        self.hidden_masks = []
        mask = np.ones((input_neuron_num, ), dtype=int)
        self.hidden_masks.append(mask)
        y = 0
        for i in range(hidden_layer_num):
            # print(i)
            hidden_layer = models.onn_layer.ONN_Layer(neuron_number=hidden_neuron_num[i], 
                                                      distance=hidden_distance[i], 
                                                      bound=hidden_bound[i], 
                                                      y=-(hidden_layer_distance[i] + y),
                                                      phi_init=phi_init,
                                                      )
            dests = np.zeros((hidden_neuron_num[i], 2), dtype=np.float64)
            dests[:, 0] = hidden_layer.x0
            #dests[:, 1] = hidden_layer.y0 + hidden_layer.h_neuron / 2
            dests[:, 1] = hidden_layer.y0
            y = hidden_layer_distance[i] + y

            mask = np.ones((hidden_neuron_num[i], ), dtype=int)
            self.hidden_masks.append(mask)

            self.layers.append(hidden_layer)
            self.dests.append(dests)
        
        # define the output plane, the output plane is defined as a passive plane with different location
        dests = np.zeros((output_neuron_num, 2), dtype=np.float64)
        dests[:, 0] = np.arange(output_bound, output_bound + output_distance * output_neuron_num, output_distance) * Const.Lambda0
        dests[:, 1] = -(output_layer_distance + y) * Const.Lambda0
        self.height = output_layer_distance + y
        self.dests.append(dests)

        self.nonlinear = nonlinear

    def remove_dummy_cells(self):
        for i in range(self.hidden_layer_num):
            layer = self.layers[i + 1]
            last_end = layer.x0[0] + layer.x0_bound
            for j in range(layer.neuron_number - 1):
                new_start = layer.x0[j + 1] - layer.x0_bound
                new_end = layer.x0[j + 1] + layer.x0_bound
                if new_start > last_end:
                    last_end = new_end
                else:
                    # remove the one with larger end
                    if new_end > last_end:
                        last_end = last_end
                        self.hidden_masks[i + 1][j + 1] = 0
                    else:
                        last_end = new_end
                        self.hidden_masks[i + 1][j] = 0
    
    def update_layer_dummy_cells(self, layer_index):
        layer = self.layers[layer_index]
        last_end = layer.x0[0] + layer.x0_bound
        for j in range(layer.neuron_number - 1):
            new_start = layer.x0[j + 1] - layer.x0_bound
            new_end = layer.x0[j + 1] + layer.x0_bound
            if new_start > last_end:
                last_end = new_end
            else:
                # remove the one with larger end
                if new_end > last_end:
                    last_end = last_end
                    self.hidden_masks[layer_index][j + 1] = 0
                else:
                    last_end = new_end
                    self.hidden_masks[layer_index][j] = 0

    def update_dests(self):
        for i in range(self.layer_num - 1):
            self.dests[i][:, 0] = self.layers[i + 1].x0
            
    def forward(self, input_Ex, verbose=False):
        """
        The forward propagation modeling of the DONN
        """
        if verbose == True:
            print("---------------- Forward Proapagation ----------------")
        # self.input_Ex.append(input_Ex)
        for layer_index, layer in enumerate(self.layers):
            # plot.plot_layer_pattern(layer, input_Ex, height=self.layer_distance / Const.Lambda0)
            if verbose == True:
                print("Proapagation from Layer %d to layer %d" % (layer_index, layer_index + 1))
            # apply mask
            input_Ex = input_Ex * self.hidden_masks[layer_index]
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            
            # self.input_Ex.append(input_Ex)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        if verbose == True:
            print("------------------------------------------------------")
        return input_Ex
    
    def loss_v4(self, input_Ex, y=None, backpropagate=True):
        """
        Loss Function of Dummy DONN
        1. calculate the forward propagation
        2. update the loss func backward
        3. remove dummy cells
        4. backward propagation
        """

        batch_size = input_Ex.shape[:-1]

        # first compute the output seprately
        Ex_out = []
        Ex_cache = []
        dx0_list = {}

        for layer_index, layer in enumerate(self.layers[:-1]):
            # apply mask
            input_Ex = input_Ex * self.hidden_masks[layer_index]
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            Ex_out.append(input_Ex)
            Ex_cache.append(cache)

        # define the location of the output plane
        output_plane = self.dests[-1]
        # Ex at output
        output_Ex_shape = batch_size + (self.output_neuron_num, )
        output_Ex = np.zeros(output_Ex_shape, dtype=np.complex64)
        output_cache = []
        input_Ex = input_Ex * self.hidden_masks[-1]
        for k in range(self.output_neuron_num):
            dest_k = output_plane[k:k + 1, :]
            Ex, cache = self.layers[-1].forward_propagation(input_Ex, dest_k)
            output_Ex[:, k:k + 1] = Ex
            output_cache.append(cache)

        if backpropagate == True:
            loss, coeff = Layers.normalized_mean_square_error(output_Ex, y, g_true=1, g_false=0, non_linear=self.nonlinear)
            # dout for last layer
            dout = [np.expand_dims(np.ones(batch_size), axis=1)] * self.output_neuron_num
            # dx0 = [[None] * self.output_neuron_num]
            for i in reversed(range(self.layer_num)):
                S = np.zeros(batch_size + (self.layers[i].neuron_number, ))
                # print("S shape:", S.shape)
                for k in range(self.output_neuron_num):
                    # neurons of last layer have different cache
                    # print("dout[%d %d] shape is: " % (i, k), dout[k].shape )
                    if i == len(self.layers) - 1:
                        cache = output_cache[k]
                    else:
                        cache = Ex_cache[i]

                    Ex_conj = np.conj(output_Ex[:, k:k + 1])
                    dout[k], dEx_dx0 = self.layers[i].backward_propagation_holomorphic_x0_v2(dout[k], cache, Ex_conj)
                    dEx_dx0 = dEx_dx0 * self.hidden_masks[i]
                    real_part = np.real(dEx_dx0)

                    S = S + coeff[:, k:k + 1] * real_part
                    #S = S + coeff[:, k:k + 1] * np.real(np.conj(output_Ex[:, k:k + 1]) * dEx_dx0)
                S = 4 * S / self.output_neuron_num
                # print("S shape:", S.shape)
                S = np.mean(S, axis=0)
                dx0_list[i] = S
            return output_Ex, loss, dx0_list

        else:
            return output_Ex

    def plot_structure(self, filename="temp"):
        input_width = self.input_bound * 2 + (self.input_neuron_num - 1) * self.input_distance
        # local_hidden_width = self.hidden_bound * 2 + (self.hidden_neuron_num - 1) * self.hidden_distance
        output_width = self.output_bound * 2 + (self.output_neuron_num - 1) * self.output_distance
        # print("Input Width: ", input_width)
        # print("Hidden Width: ", hidden_width)
        # print("Output Width: ", output_width)
        # print("Height: ", self.height)

        fig, ax = pyplot.subplots()
        for layer_index, layer in enumerate(self.layers):
            # plot.plot_layer_pattern(layer, input_Ex, height=self.layer_distance / Const.Lambda0)
            # print("Proapagation from Layer %d to layer %d" % (layer_index, layer_index + 1))
            x = layer.x0 / Const.Lambda0
            # print(x)
            y = layer.y0 * np.ones(x.shape) / Const.Lambda0
            # print(y)
            ax.scatter(x, y)
        ax.scatter(self.dests[-1][:, 0] / Const.Lambda0, self.dests[-1][:, 1] / Const.Lambda0)
        # print(self.dests[-1][:, 0] / Const.Lambda0)
        #ax.set_xlim(0, input_width)
        #ax.set_ylim(0, self.height)
        ax.axis('equal')
        pyplot.savefig("./figures/" + filename + "_structure.pdf", format='pdf', bbox_inches='tight')
        # pyplot.show()

def get_local_global_donn_example(input_neuron_num=196,
                                output_neuron_num=10,
                                local_layer_num=2,
                                local_neuron_num=1000,
                                local_neuron_distance=1.5e-6 / Const.Lambda0,
                                local_layer_distance=15e-6 / Const.Lambda0,
                                global_layer_num=3,
                                global_neuron_num=1000,
                                global_neuron_distance=1.5e-6 / Const.Lambda0,
                                global_layer_distance=1500e-6 / Const.Lambda0,
                                input_distance=7.6e-6 / Const.Lambda0,
                                output_distance=150e-6 / Const.Lambda0,
                                phi_init="default",
                                first_layer_distance=30e-6 / Const.Lambda0,
                                nonlinear=False):
    """
    Get Flexible DONN Example

    Local Layer Num: 2
    Local Layer Distance: 15 um

    Global Layer Num: 3
    Global Layer Distance: 1500 um

    Input Neuron Distance: 7.6 um
    Local / Global Neuron Distance: 1.5 um
    Output Neuron Distance: 150 um
    """

    
    ## construct the deep onn structure
    # define the structure parameters

    # new_size = 10
    #hidden_neuron_num = 100
    #output_neuron_num = 10
    # hidden_layer_num = 1
    # layer_distance = hidden_distance * hidden_neuron_num



    hidden_layer_num = local_layer_num + global_layer_num
    hidden_neuron_num_list = [local_neuron_num] * local_layer_num + [global_neuron_num] * global_layer_num
    hidden_neuron_distance_list = [local_neuron_distance] * local_layer_num + [global_neuron_distance] * global_layer_num
    hidden_layer_distance_list = [local_layer_distance] * local_layer_num + [global_layer_distance] * global_layer_num

    # construct first layer distance
    hidden_layer_distance_list[0] = first_layer_distance

    total_width = max((input_neuron_num + 10) * input_distance, 
                        (local_neuron_num + 10) * local_neuron_distance, 
                        (global_neuron_num + 10) * global_neuron_distance) 

    input_bound = utils.helpers.get_bound(input_neuron_num, input_distance, total_width)
    local_bound = utils.helpers.get_bound(local_neuron_num, local_neuron_distance, total_width)
    global_bound = utils.helpers.get_bound(global_neuron_num, global_neuron_distance, total_width)
    output_bound = utils.helpers.get_bound(output_neuron_num, output_distance, total_width)

    hidden_bound_list = [local_bound] * local_layer_num + [global_bound] * global_layer_num
    #print(input_Ex)

    test_donn = Dummy_DONN(input_neuron_num=input_neuron_num,
                            input_distance=input_distance,
                            input_bound=input_bound,
                            output_neuron_num=output_neuron_num,
                            output_distance=output_distance,
                            output_bound=output_bound,
                            hidden_layer_num=hidden_layer_num,
                            hidden_neuron_num=hidden_neuron_num_list,
                            hidden_distance=hidden_neuron_distance_list,
                            hidden_bound=hidden_bound_list,
                            hidden_layer_distance=hidden_layer_distance_list,
                            output_layer_distance=global_layer_distance,
                            phi_init=phi_init,
                            nonlinear=nonlinear
                            )

    return test_donn

def test_remove_dummy_cells():

    # define data
    new_size = 14
    (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(new_size)

    # generate dummy donn with close examples
    hidden_neuron_distance = Const.x0_bound * 2
    input_neuron_num = new_size ** 2
    input_distance = 2
    hidden_layer_num = 2
    hidden_neuron_num = 400
    hidden_layer_distance = 800
    output_layer_distance = 400
    output_neuron_num = 10
    output_distance = 30
    
    # make sure input layer widest
    total_width = (input_neuron_num + 10) * input_distance
    input_bound = utils.helpers.get_bound(input_neuron_num, input_distance, total_width)
    hidden_bound = utils.helpers.get_bound(hidden_neuron_num, hidden_neuron_distance, total_width)
    output_bound = utils.helpers.get_bound(output_neuron_num, output_distance, total_width)

    hidden_neuron_distance_list = [hidden_neuron_distance] * hidden_layer_num
    hidden_neuron_num_list = [hidden_neuron_num] * hidden_layer_num
    hidden_layer_distance_list = [hidden_layer_distance] * hidden_layer_num 
    hidden_bound_list = [hidden_bound] * hidden_layer_num

    test_donn = Dummy_DONN(input_neuron_num=input_neuron_num,
                            input_distance=input_distance,
                            input_bound=input_bound,
                            output_neuron_num=output_neuron_num,
                            output_distance=output_distance,
                            output_bound=output_bound,
                            hidden_layer_num=hidden_layer_num,
                            hidden_neuron_num=hidden_neuron_num_list,
                            hidden_distance=hidden_neuron_distance_list,
                            hidden_bound=hidden_bound_list,
                            hidden_layer_distance=hidden_layer_distance_list,
                            output_layer_distance=output_layer_distance,
                            phi_init="default",
                            nonlinear=False
                            )

    # add random move to x0
    for i in range (test_donn.hidden_layer_num):
        test_donn.layers[i + 1].x0 += np.random.rand(test_donn.layers[i + 1].neuron_number) * (test_donn.layers[i + 1].x0_bound / 5) - test_donn.layers[i + 1].x0_bound / 10

    test_donn.plot_structure()

    # remove dummy cells
    test_donn.remove_dummy_cells()

    # print new onn layer x0
    for i in range(len(test_donn.hidden_masks)):
        print("layer %d masks:" % i)
        print(test_donn.hidden_masks[i])

    # check overlap
    for i in range(len(test_donn.hidden_masks)):
        layer = test_donn.layers[i]
        for j in range(layer.neuron_number - 1):
            if (layer.x0[j] + layer.x0_bound > layer.x0[j + 1] - layer.x0_bound):
                assert ValueError("Overlap Occur!")

    # count nonoverlap number
    for i in range(len(test_donn.hidden_masks)):
        print("active neuron number in layer %d: " % i, np.count_nonzero(test_donn.hidden_masks[i]))

    # test forward
    input_Ex = train_X[0:1]
    print(input_Ex)
    output_Ex = test_donn.forward(input_Ex, verbose=True)
    print(output_Ex)

    # test focus network
    for layer_index, layer in enumerate(test_donn.layers[:-1]):
        test_donn.hidden_masks[layer_index] = np.ones((layer.neuron_number, ), dtype=int)
    iter_num = 300
    input_Ex = np.ones((1, input_neuron_num), dtype=int)
    target = np.zeros(1, dtype=int)
    target[0] = 9
    lr = 1e-11
    for iter in range(iter_num): 
        output_Ex, loss, dx0_list = test_donn.loss_v4(input_Ex, y=target)
        for i in range(test_donn.layer_num):
            dx = lr * dx0_list[i]
            test_donn.layers[i].x0 -= dx
        
        test_donn.update_dests()
        test_donn.remove_dummy_cells()
        output_Ex = test_donn.forward(input_Ex)
        if iter == (iter_num - 1): 
            print("Iter %d Loss:" % iter, loss)
            print("Output Ex:", np.abs(output_Ex))

    # count nonoverlap number
    neuron_number = []
    for i in range(len(test_donn.hidden_masks)):
        non_zero = np.count_nonzero(test_donn.hidden_masks[i])
        print("active neuron number in layer %d: " % i, non_zero)
        if i != 0:
            neuron_number.append(non_zero)

    # compare with not dummy cell 
    # generate compare donn with close examples
    hidden_neuron_distance_list = []
    hidden_bound_list = []
    hidden_layer_distance_list = []
    hidden_neuron_num_list = neuron_number

    input_neuron_num = new_size ** 2
    input_distance = 2
    hidden_layer_num = 2
    # a list

    output_layer_distance = 400
    output_neuron_num = 10
    output_distance = 30
    
    total_width = (input_neuron_num + 10) *     
    input_bound = utils.helpers.get_bound(input_neuron_num, input_distance, total_width)
    for i in range(hidden_layer_num):
        distance = input_neuron_num * input_distance / hidden_neuron_num_list[i]
        hidden_neuron_distance_list.append(distance)
        hidden_bound_list.append(utils.helpers.get_bound(hidden_neuron_num_list[i], distance, total_width))
        hidden_layer_distance_list.append(800)

    output_bound = utils.helpers.get_bound(output_neuron_num, output_distance, total_width)

    comp_donn = Flexible_DONN(input_neuron_num=input_neuron_num,
                            input_distance=input_distance,
                            input_bound=input_bound,
                            output_neuron_num=output_neuron_num,
                            output_distance=output_distance,
                            output_bound=output_bound,
                            hidden_layer_num=hidden_layer_num,
                            hidden_neuron_num=hidden_neuron_num_list,
                            hidden_distance=hidden_neuron_distance_list,
                            hidden_bound=hidden_bound_list,
                            hidden_layer_distance=hidden_layer_distance_list,
                            output_layer_distance=output_layer_distance,
                            phi_init="default",
                            nonlinear=False
                            )

    # test focus network
    iter_num = 300
    input_Ex = np.ones((1, input_neuron_num), dtype=int)
    target = np.zeros(1, dtype=int)
    target[0] = 9
    lr = 1e-11
    for iter in range(iter_num): 
        output_Ex, loss, dx0_list = comp_donn.loss_v4(input_Ex, y=target)
        for i in range(comp_donn.layer_num):
            dx = lr * dx0_list[i]
            comp_donn.layers[i].x0 -= dx
        
        comp_donn.update_dests()
        output_Ex = comp_donn.forward(input_Ex)
        if iter == (iter_num - 1): 
            print("Iter %d Loss:" % iter, loss)
            print("Output Ex:", np.abs(output_Ex))


def main():
    size = 14
    donn = get_local_global_donn_example(input_neuron_num=size ** 2,
                                local_neuron_distance=1.5e-6 / Const.Lambda0,
                                global_neuron_distance=1.5e-6 / Const.Lambda0,
                                input_distance=7.6e-6 / Const.Lambda0,
                                output_distance=150e-6 / Const.Lambda0,
                                local_layer_distance=15e-6 / Const.Lambda0,
                                )
    
    donn.plot_structure()
    
if __name__ == '__main__':
    test_remove_dummy_cells()