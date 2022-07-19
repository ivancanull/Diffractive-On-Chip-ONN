from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
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


class DONN(object):
    """
    Deep Optical Neural Network Model
    Attributes: 

        input_batch_num: marked as m, the input Ex0 is a batch of many images, for example,
                         therefore, we can handle multiple images training.
        input_Ex: the input Ex, shape is (m, input_n)
        neuron_num: neuron number
        bound: boundary distance between the first neuron and the left limit,
               also the distance between the last and the right limit
        distance: distance between neighbour neurons
        output_Ex: the output Ex, shape is (m, output_n)
    """

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
                 layer_distance,
                 phi_init="default",
                 nonlinear=False,
                 ) -> None:
        super().__init__()
        
        self.layers = []
        self.dests = []
        self.Ex_out = []
        self.Ex_cache = []
        
        self.dw = []
        
        self.layer_num = hidden_layer_num + 1

        # define input layers
        self.layer_distance = layer_distance * Const.Lambda0

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

        # initialize input Ex
        # self.input_layer.Ex0 = input_Ex

        # define hidden layers
        self.layers.append(input_layer)

        for i in range(hidden_layer_num):
            
            hidden_layer = models.onn_layer.ONN_Layer(neuron_number=hidden_neuron_num, 
                                                      distance=hidden_distance, 
                                                      bound=hidden_bound, 
                                                      y=-layer_distance * (i + 1),
                                                      phi_init=phi_init,
                                                      )
            dests = np.zeros((hidden_neuron_num, 2), dtype=np.float64)
            dests[:, 0] = hidden_layer.x0
            #dests[:, 1] = hidden_layer.y0 + hidden_layer.h_neuron / 2
            dests[:, 1] = hidden_layer.y0

            self.layers.append(hidden_layer)
            self.dests.append(dests)                                                            

        # define the output plane, the output plane is defined as a passive plane with different location
        dests = np.zeros((output_neuron_num, 2), dtype=np.float64)
        dests[:, 0] = np.arange(output_bound, output_bound + output_distance * output_neuron_num, output_distance) * Const.Lambda0
        dests[:, 1] = -layer_distance * (hidden_layer_num + 1) * Const.Lambda0
        self.height = layer_distance * (hidden_layer_num + 1)
        self.dests.append(dests)

        self.nonlinear = nonlinear
        """
        
        # define the output Ex
        self.output_Ex = np.zeros((input_batch_num, output_neuron_num), dtype=np.float64)
        print("Output layer: ", self.output_Ex.shape)

        """
    
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
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            # self.input_Ex.append(input_Ex)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        if verbose == True:
            print("------------------------------------------------------")
        return input_Ex

    def loss(self, input_Ex, y=None, backpropagate=True):
        """
        Evaluate Loss and Gradient for the DONN Model
        """
        dout_r_list = {}
        dout_i_list = {}
        dh_list = {}

        self.Ex_out.append(input_Ex)
        for layer_index, layer in enumerate(self.layers):
            # plot.plot_layer_pattern(layer, input_Ex, height=self.layer_distance / Const.Lambda0)
            # print("Proapagation from Layer %d to layer %d" % (layer_index, layer_index + 1))
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            self.Ex_out.append(input_Ex)
            self.Ex_cache.append(cache)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        # print("------------------------------------------------------")
        
        # normaliztion layer
        Ex_norm, norm_cache = Layers.norm_layer_forward(input_Ex)
        self.Ex_out.append(Ex_norm)
        self.Ex_cache.append(norm_cache)
        inference = np.argmax(Ex_norm, axis=1)

        # print("dout shape: ", dout.shape)

        if backpropagate == True:
            # calculate_loss
            loss, dout = Layers.softmax_loss(Ex_norm, y)
            # do backpropagate
            dout_r, dout_i = Layers.norm_layer_backward(dout, norm_cache)

            for i in reversed(range(len(self.layers))):
                cache = self.Ex_cache[i]
                dout_r, dout_i, dh = self.layers[i].backward_propagation(dout_r, dout_i, cache)
                dout_r_list[i] = dout_r
                dout_i_list[i] = dout_i
                dh_list[i] = dh

            return inference, loss, dout_r_list, dout_i_list, dh_list
        
        else:
            return inference
        
    def loss_v2(self, input_Ex, y=None, backpropagate=True):
        
        """
        Ref:

        X. Lin et al., “All-optical machine learning using diffractive 
        deep neural networks,” Science (80-. )., vol. 361, no. 6406, 
        pp. 1004–1008, 2018, doi: 10.1126/science.aat8084.

        Evaluate Loss and Gradient according to the following equation:

        s_k = abs(out_k) ^ 2
        g_k = target of k

        Loss: mean square error
        L = the average of (s_k - g_k) ^ 2

        dEx = 4 / K * sum[(s_k - g_k) * real(out_k.conj() * dEx_dphi)
                                 shape:        (n, 1)       (1, m_i)


        """

        batch_size = input_Ex.shape[:-1]

        # first compute the output seprately
        Ex_out = []
        Ex_cache = []
        dphi_list = {}

        for layer_index, layer in enumerate(self.layers[:-1]):
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            Ex_out.append(input_Ex)
            Ex_cache.append(cache)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        # print("------------------------------------------------------")        
        # normaliztion layer

        # define the location of the output plane
        output_plane = self.dests[-1]
        # Ex at output
        output_Ex_shape = batch_size + (self.output_neuron_num, )
        output_Ex = np.zeros(output_Ex_shape, dtype=np.complex64)
        output_cache = []
        for k in range(self.output_neuron_num):
            dest_k = output_plane[k:k + 1, :]
            Ex, cache = self.layers[-1].forward_propagation(input_Ex, dest_k)
            output_Ex[:, k:k + 1] = Ex
            output_cache.append(cache)


        if backpropagate == True:
            loss, s_minus_g = Layers.mean_square_error(output_Ex, y, g_true=1000, g_false=50)
            # dout for last layer
            dout = [np.expand_dims(np.ones(batch_size), axis=1)] * self.output_neuron_num
            dphi = [[None] * self.output_neuron_num]
            for i in reversed(range(self.layer_num)):
                S = np.zeros(batch_size + (self.layers[i].neuron_number, ))
                for k in range(self.output_neuron_num):
                    # neurons of last layer have different cache
                    # print("dout[%d %d] shape is: " % (i, k), dout[k].shape )
                    if i == len(self.layers) - 1:
                        cache = output_cache[k]
                    else:
                        cache = Ex_cache[i]
                    dout[k], dEx_dphi = self.layers[i].backward_propagation_holomorphic_phi(dout[k], cache)
                    
                    # dEx_dphi shape (1, m_i)
                    dEx_dphi = np.sum(dEx_dphi, axis=0, keepdims=True)
                    # print("dEx_dphi[%d %d] shape is: " % (i, k), dEx_dphi.shape )
                    # S shape (n, m_i)
                    S = S + s_minus_g[:, k:k + 1] * np.real(np.conj(output_Ex[:, k:k + 1]) * dEx_dphi)
                S = 4 * S / self.output_neuron_num
                S = np.mean(S, axis=0)
                dphi_list[i] = S
                # print("S shape: ", S.shape)
            return output_Ex, loss, dphi_list

        else:
            return output_Ex
    
        # backpropagation
        # for k in range(self.output_neuron_num):


    def loss_v3(self, input_Ex, y=None, backpropagate=True):
        
        """
        Phase-Modulated Loss Function

        Ref: T. Fu et al., “On-chip photonic diffractive optical neural 
        network based on a spatial domain electromagnetic propagation model,” 
        Opt. Express, vol. 29, no. 20, p. 31924, 2021, doi: 10.1364/oe.435183.

        Evaluate Loss and Gradient according to the following equation:

        s_k = abs(out_k) ^ 2
        g_k = target of k

        Loss: normalized mean square error
        L = the average of (s_k / S - g_k) ^ 2
        S = sum(s_k)
        dEx = 4 / K * sum[(s_k / S - g_k) * (S - s_k) / (S ^ 2) * 
              real(out_k.conj() * dEx_dphi)]
        shape:       (n, 1)       (1, m_i)


        """

        batch_size = input_Ex.shape[:-1]

        # first compute the output seprately
        Ex_out = []
        Ex_cache = []
        dphi_list = {}

        for layer_index, layer in enumerate(self.layers[:-1]):
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            Ex_out.append(input_Ex)
            Ex_cache.append(cache)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        # print("------------------------------------------------------")        
        # normaliztion layer 

        # define the location of the output plane
        output_plane = self.dests[-1]
        # Ex at output
        output_Ex_shape = batch_size + (self.output_neuron_num, )
        output_Ex = np.zeros(output_Ex_shape, dtype=np.complex64)
        output_cache = []
        for k in range(self.output_neuron_num):
            dest_k = output_plane[k:k + 1, :]
            Ex, cache = self.layers[-1].forward_propagation(input_Ex, dest_k)
            output_Ex[:, k:k + 1] = Ex
            output_cache.append(cache)


        if backpropagate == True:
            loss, coeff = Layers.normalized_mean_square_error(output_Ex, y, g_true=1, g_false=0, non_linear=self.nonlinear)
            # dout for last layer
            dout = [np.expand_dims(np.ones(batch_size), axis=1)] * self.output_neuron_num
            dphi = [[None] * self.output_neuron_num]
            for i in reversed(range(self.layer_num)):
                S = np.zeros(batch_size + (self.layers[i].neuron_number, ))
                for k in range(self.output_neuron_num):
                    # neurons of last layer have different cache
                    # print("dout[%d %d] shape is: " % (i, k), dout[k].shape )
                    if i == len(self.layers) - 1:
                        cache = output_cache[k]
                    else:
                        cache = Ex_cache[i]
                    dout[k], dEx_dphi = self.layers[i].backward_propagation_holomorphic_phi(dout[k], cache)
                    
                    # dEx_dphi shape (1, m_i)
                    dEx_dphi = np.sum(dEx_dphi, axis=0, keepdims=True)
                    # print("dEx_dphi[%d %d] shape is: " % (i, k), dEx_dphi.shape )
                    # S shape (n, m_i)
                    # print("s_sum shape:", coeff.shape)
                    S = S + coeff[:, k:k + 1] * np.real(np.conj(output_Ex[:, k:k + 1]) * dEx_dphi)
                S = 4 * S / self.output_neuron_num
                S = np.mean(S, axis=0)
                dphi_list[i] = S
                # print("S shape: ", S.shape)
            return output_Ex, loss, dphi_list

        else:
            return output_Ex
    
    def loss_v4(self, input_Ex, y=None, backpropagate=True):
        
        """
        Location-Modulated Loss Function

        Evaluate Loss and Gradient according to the following equation:

        s_k = abs(out_k) ^ 2
        g_k = target of k

        Loss: normalized mean square error
        L = the average of (s_k / S - g_k) ^ 2
        S = sum(s_k)
        dEx = 4 / K * sum[(s_k / S - g_k) * (S - s_k) / (S ^ 2) * 
              real(out_k.conj() * dEx_dphi)]
        shape:       (n, 1)       (1, m_i)


        """

        batch_size = input_Ex.shape[:-1]

        # first compute the output seprately
        Ex_out = []
        Ex_cache = []
        dx0_list = {}

        for layer_index, layer in enumerate(self.layers[:-1]):
            input_Ex, cache = layer.forward_propagation(input_Ex, self.dests[layer_index])
            Ex_out.append(input_Ex)
            Ex_cache.append(cache)
            # np.savetxt("./temp/layer_" + str(layer_index) + ".txt", np.abs(input_Ex))
        # print("------------------------------------------------------")        
        # normaliztion layer 

        # define the location of the output plane
        output_plane = self.dests[-1]
        # Ex at output
        output_Ex_shape = batch_size + (self.output_neuron_num, )
        output_Ex = np.zeros(output_Ex_shape, dtype=np.complex64)
        output_cache = []
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

                    method = 2

                    if method == 1:
                        # method 1
                        dout[k], dEx_dx0 = self.layers[i].backward_propagation_holomorphic_x0(dout[k], cache)
                        output_expanded = np.expand_dims(np.conj(output_Ex[:, k:k + 1]), axis=-1)
                        real_part = np.sum(np.real(output_expanded * dEx_dx0), axis=-1)
                    else:
                        # method 2
                        Ex_conj = np.conj(output_Ex[:, k:k + 1])
                        dout[k], dEx_dx0 = self.layers[i].backward_propagation_holomorphic_x0_v2(dout[k], cache, Ex_conj)
                        real_part = np.real(dEx_dx0)
                    # dEx_dphi shape (1, m_i)
                    
                    # dEx_dx0 = np.mean(dEx_dx0, axis=0, keepdims=True)
                    # print("dEx_dx0 shape:", dEx_dx0.shape)
                    # print("dEx_dx0 shape: ", dEx_dx0.shape)
                    # print("dEx_dphi[%d %d] shape is: " % (i, k), dEx_dx0.shape )
                    # S shape (n, m_i)
                    # print("s_sum shape:", coeff.shape)
                    # print(coeff[:, k:k + 1] * np.real(np.conj(output_Ex[:, k:k + 1]) * dEx_dx0))

                    # dout shape (n, m_i, m_o)
                    # print("dx0 shape: ", dEx_dx0.shape)
                    # print("output_expanded shape: ", output_expanded.shape)
                    # print("real_part shape:", real_part.shape)
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
        hidden_width = self.hidden_bound * 2 + (self.hidden_neuron_num - 1) * self.hidden_distance
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
            y = layer.y0 * np.ones(x.shape) / Const.Lambda0
            ax.scatter(x, y)
        ax.scatter(self.dests[-1][:, 0] / Const.Lambda0, self.dests[-1][:, 1] / Const.Lambda0)
        #ax.set_xlim(0, input_width)
        #ax.set_ylim(0, self.height)
        ax.axis('equal')
        pyplot.savefig("./figures/" + filename + "_structure.pdf", format='pdf', bbox_inches='tight')
        # pyplot.show()

def get_donn_example(input_neuron_num=25,
                     hidden_layer_num=1,
                     hidden_neuron_num=30,
                     output_neuron_num=10,
                     input_distance=30,
                     hidden_distance=4,
                     output_distance=160,
                     phi_init="default",
                     layer_distance=1600,
                     nonlinear=False):
    ## construct the deep onn structure
    # define the structure parameters

    # new_size = 10
    #hidden_neuron_num = 100
    #output_neuron_num = 10
    assert hidden_neuron_num > 20
    # hidden_layer_num = 1
    layer_distance = hidden_distance * hidden_neuron_num
    total_width = max((input_neuron_num + 10) * input_distance, hidden_distance * (hidden_neuron_num + 10)) 

    # hidden_distance = 12
    # for small input size
    # input_distance = hidden_distance * max(1, int(hidden_neuron_num / input_neuron_num))
    # output_distance = hidden_distance * max(1, (int(hidden_neuron_num / output_neuron_num) - 0.5))
    
    # input_distance = hidden_distance
    # output_distance = hidden_distance * 80

    input_bound = utils.helpers.get_bound(input_neuron_num, input_distance, total_width)
    hidden_bound = utils.helpers.get_bound(hidden_neuron_num, hidden_distance, total_width)
    output_bound = utils.helpers.get_bound(output_neuron_num, output_distance, total_width)

    #print(input_Ex)

    test_donn = DONN(input_neuron_num=input_neuron_num,
                     input_distance=input_distance,
                     input_bound=input_bound,
                     output_neuron_num=output_neuron_num,
                     output_distance=output_distance,
                     output_bound=output_bound,
                     hidden_layer_num=hidden_layer_num,
                     hidden_neuron_num=hidden_neuron_num,
                     hidden_distance=hidden_distance,
                     hidden_bound=hidden_bound,
                     layer_distance=layer_distance,
                     phi_init=phi_init,
                     nonlinear=nonlinear
                     )
    
    return test_donn

def test_plot_structure():
    test_donn = get_donn_example(new_size=8, hidden_layer_num=4)
    test_donn.plot_structure()


def main():
    ## construct the deep onn structure
    # define the structure parameters

    test_plot_structure()
    """    
    test_donn = get_donn_example()

    example_num = 4
    new_size = 10

    example_input = encoding.utils.get_MNIST_example(example_num)
    compressed_input = np.zeros((example_input.shape[0], new_size, new_size), dtype=np.float64)
    for i in range(example_num):
        compressed_input[i] = cv2.resize(example_input[i], (new_size, new_size))
    input_Ex = np.reshape(compressed_input, (example_num, new_size ** 2))
    np.savetxt("input.txt", np.abs(input_Ex))

    forward = test_donn.forward(input_Ex)
    np.savetxt("result.txt", np.abs(forward))
    print(test_donn.input_Ex[0].shape)
    #plot.plot_layer_pattern(test_donn.layers[1], test_donn.input_Ex[1][0:1], height=test_donn.layer_distance / Const.Lambda0)

    """
if __name__ == '__main__':
    main()