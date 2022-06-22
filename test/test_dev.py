"""
This file tests focus network
"""
from email.policy import default
import sys, os

from torch import mode

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

from utils import constants as Const
#import utils.helpers
#import utils.plot as plot
from models.onn_layer import ONN_Layer
import models.layers as Layers
import models.donn as donn
#from models.onn_layer import ONN_Layer
#from solver import Solver 

import encoding.utils
#import cv2
def main():
    
    test_backward_dev()

def test_backward_dev():
    batch_size = 4
    new_size = 7
    (compressed_Ex, sampled_y) = encoding.utils.get_training_example(batch_size, new_size, False)
    
    neuron_number = new_size ** 2
    test_layer = ONN_Layer(neuron_number=neuron_number)
    # test following layer structure
    hidden_bound = 10 
    hidden_num = 10
    hidden_distance = 6
    hidden_spacing = 20

    # define the dests coords
    dests = np.zeros((hidden_num, 2), dtype=np.float64)
    dests[:, 0] = np.arange(hidden_bound, hidden_bound + hidden_distance * hidden_num, hidden_distance) * Const.Lambda0
    dests[:, 1] = np.full(hidden_num, fill_value=-hidden_spacing * Const.Lambda0)

    Ex, cache = test_layer.forward_propagation(compressed_Ex, dests)

    # define the location of the output plane
    output_plane = dests
    # Ex at output
    output_Ex_shape = (batch_size, hidden_num, )
    output_Ex = np.zeros(output_Ex_shape, dtype=np.complex64)
    S = np.zeros([batch_size, neuron_number, ])
    for k in range(hidden_num):
        dest_k = output_plane[k:k + 1, :]
        Ex, cache = test_layer.forward_propagation(compressed_Ex, dest_k)
        output_Ex[:, k:k + 1] = Ex
        loss, coeff = Layers.normalized_mean_square_error(output_Ex, sampled_y, g_true=1, g_false=0,)

        dout = np.expand_dims(np.ones(batch_size), axis=1)


        Ex_conj = np.conj(output_Ex[:, k:k + 1])
        _, dEx_dx0 = test_layer.backward_propagation_holomorphic_x0_v2(dout, cache, Ex_conj)
        real_part = np.real(dEx_dx0)
        print(real_part)
        S = S + coeff[:, k:k + 1] * real_part
        S = 4 * S / hidden_num
        S = np.mean(S, axis=0)
    print(S)


def test_multilayer_dev(iter, mode="phi", plot=False):

    """
    Test Multilayer Focus Network with different modes: 
  
        phi: phase-modulated
        x0: location-modulated

    """
    if mode == "phi":
        phi_init = "random"
    elif mode == "x0":
        phi_init = "default"

    DONN_model = donn.DONN(input_neuron_num=12, input_distance=12, input_bound=40,
                           output_neuron_num=10, output_distance=12, output_bound=40,
                           hidden_layer_num=1,
                           hidden_neuron_num=10, hidden_distance=12, hidden_bound=40,
                           layer_distance=200,
                           phi_init=phi_init)
    
    dout = np.ones((1, 10))
    target = np.zeros(1, dtype=np.int8)

    target_index = 0
    target[0] = target_index
    for k in range(iter):
        input_Ex = np.ones((1, 12)) * 1e-5
        out = DONN_model.forward(input_Ex)

        s = np.zeros(10)
        if mode == "phi":
            output_Ex, loss, dx_list = DONN_model.loss_v3(input_Ex, y=target)
        elif mode == "x0":
            output_Ex, loss, dx_list = DONN_model.loss_v4(input_Ex, y=target)
        if iter <= 20:
            print("Ex:", output_Ex)
            print("abs:", np.abs(output_Ex))
            print("loss:", loss)
        elif k % 100 == 0:
            print("iter %d Ex norm:" % k, np.abs(output_Ex))
            print("iter %d loss:" % k, np.mean(loss))
        lr = {}
        if mode == "phi":
            lr[0] = 1e-2
            lr[1] = 1e-2
        elif mode == "x0":
            lr[0] = 1e-12
            lr[1] = 1e-11

        for i in range(DONN_model.layer_num):

            dx = lr[i] * dx_list[i]
            
            
            if mode == "phi":
                if iter <= 20:
                    print("dphi", i, ":", dx)
                DONN_model.layers[i].phi -= dx
                DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
                DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi
            elif mode == "x0":
                if iter <= 20:
                    print("dx0", i, ":", dx / Const.Lambda0)
                DONN_model.layers[i].x0 -= dx
                x0_outbound_l = DONN_model.layers[i].x0 < DONN_model.layers[i].x0_left_limit
                x0_outbound_r = DONN_model.layers[i].x0 > DONN_model.layers[i].x0_right_limit
                DONN_model.layers[i].x0[x0_outbound_l] = DONN_model.layers[i].x0_left_limit[x0_outbound_l]
                DONN_model.layers[i].x0[x0_outbound_r] = DONN_model.layers[i].x0_right_limit[x0_outbound_r]

        input_Ex = np.ones((1, 12)) * 1e-5
        out = DONN_model.forward(input_Ex)
        #print("out_r: ", np.abs(out[0, 0]), np.abs(out[0, 4]), np.abs(out[0, 9]))
    # print("out_r: ", np.abs(out[0, target_index]))
    if plot:
        DONN_model.plot_structure()



# Important:
# 
# Use the function above instead.
# Do not use this function !!
#
def test_dev(mode):
    DONN_model = donn.DONN(input_neuron_num=12, input_distance=12, input_bound=40,
                           output_neuron_num=10, output_distance=12, output_bound=40,
                           hidden_layer_num=1,
                           hidden_neuron_num=10, hidden_distance=12, hidden_bound=40,
                           layer_distance=200)
    input_Ex = np.ones((1, 12)) * 1e-5

    output_Ex, loss, dphi_list = DONN_model.loss_v2(input_Ex)


    for layer_index, layer in enumerate(DONN_model.layers):
        # plot.plot_layer_pattern(layer, input_Ex, height=self.layer_distance / Const.Lambda0)
        # print("Proapagation from Layer %d to layer %d" % (layer_index, layer_index + 1))
        input_Ex, cache = layer.forward_propagation(input_Ex, DONN_model.dests[layer_index])
        DONN_model.Ex_out.append(input_Ex)
        DONN_model.Ex_cache.append(cache)

    dout_dic = {}
    dphi_dic = {}
    dx0_dic = {}
    #dout = np.zeros(10)
    #dout[0] = 1
    iter = 1000
    lr = 1e-2
    for k in range(iter):

        input_Ex = np.ones((1, 12))
        for layer_index, layer in enumerate(DONN_model.layers):
            # plot.plot_layer_pattern(layer, input_Ex, height=self.layer_distance / Const.Lambda0)
            # print("Proapagation from Layer %d to layer %d" % (layer_index, layer_index + 1))
            input_Ex, cache = layer.forward_propagation(input_Ex, DONN_model.dests[layer_index])
            DONN_model.Ex_out.append(input_Ex)
            DONN_model.Ex_cache.append(cache)
        
        dout = np.ones((1, 10))
        target = np.zeros(10)

        target_index = 5
        target[target_index] = 1
        s = np.zeros(10)
        for i in reversed(range(len(DONN_model.layers))):
            cache = DONN_model.Ex_cache[i]

            if mode == "phi":
                dout, dphi = DONN_model.layers[i].backward_propagation_holomorphic_phi(dout, cache)
                dout_dic[i] = dout
                dphi_dic[i] = dphi
                dEx_dphi = 0
                for j in range(10):
                    if j == target_index: 
                        s[j] = np.abs(input_Ex[0, j]) ** 2
                        dEx_dphi += (s[j] - target[j]) * np.real(np.conj(input_Ex[0, j]) * dphi[j])
                # print(dEx_dphi)
                DONN_model.layers[i].phi -= lr * dEx_dphi
                DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
                DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi
            elif mode == "x0":
                dout, dx0 = DONN_model.layers[i].backward_propagation_holomorphic_x0(dout, cache)
                dout_dic[i] = dout
                dx0_dic[i] = dx0
                dEx_dx0 = 0
                for j in range(10):
                    if j == target_index: 
                        s[j] = np.abs(input_Ex[0, j]) ** 2
                        dEx_dx0 += (s[j] - target[j]) * np.real(np.conj(input_Ex[0, j]) * dx0[j])
                # print(lr * dEx_dx0 / Const.Lambda0)
                DONN_model.layers[i].x0 -= lr * dEx_dx0
                x0_outbound_l = DONN_model.layers[i].x0 < DONN_model.layers[i].x0_left_limit
                x0_outbound_r = DONN_model.layers[i].x0 > DONN_model.layers[i].x0_right_limit
                DONN_model.layers[i].x0[x0_outbound_l] = DONN_model.layers[i].x0_left_limit[x0_outbound_l]
                DONN_model.layers[i].x0[x0_outbound_r] = DONN_model.layers[i].x0_right_limit[x0_outbound_r]
        input_Ex = np.ones((1, 12))
        
        #print("final: ", np.abs(out))
        out = DONN_model.forward(input_Ex)
        print("out_r: ", np.abs(out[0, target_index]))


if __name__ == '__main__':
    main()
    pass

