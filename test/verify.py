import sys, os

from pandas import test
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import pandas as pd
import csv

from models.onn_layer import ONN_Layer
import utils.plot
from utils import constants as Const
from matplotlib import pyplot as plt



def test_plot():
    # define neuron number
    neuron_num = 2
    # construct an ONN layer
    onn_layer = ONN_Layer(neuron_number=neuron_num,)
    # define input Ex
    input_Ex = np.ones((1, neuron_num))
    utils.plot.plot_layer_pattern(layer=onn_layer, input_Ex=input_Ex, figname="./figures/test_plot.pdf", height=200e-6 / Const.Lambda0, mode="norm")

def verify():

    truth_data = pd.read_csv('./data/source_base.csv', header=None)
    # print(truth_data)
    plt.plot(truth_data[0], truth_data[1], label='simulated')

    # define neuron number
    neuron_num = 1
    # construct an ONN layer
    onn_layer = ONN_Layer(neuron_number=neuron_num,)
    # define input Ex
    input_Ex = np.ones((1, neuron_num))

    print(onn_layer.x0)

    L_area = 25 * Const.Lambda0
    H_area = 15 * Const.Lambda0
    sample_num = truth_data.shape[0]
    print(sample_num)
    
    # define dests location
    dests = np.zeros((sample_num, 2))
    dests[:, 0] = np.linspace(-L_area / 2, L_area / 2, sample_num) + onn_layer.x0
    # dests[:, 0] = truth_data[0] * 1e-6 + onn_layer.x0
    dests[:, 1] = - H_area / 2 - Const.Lambda0
    
    # calculate modeled results
    x, y = onn_layer.convert_dests(dests)
    #Ex = onn_layer.Ex_input(x, y)
    #Ex = Ex.reshape((sample_num, ))

    plt.plot((dests[:, 0] - onn_layer.x0) / 1e-6, np.abs(Ex), label='modeled')
    # plt.plot(truth_data[0], np.abs(Ex), label='modeled')





    plt.xlabel('x(um)')
    plt.ylabel('Enorm')
    plt.title('Verification for gauss beam')
    plt.legend()
    
    plt.savefig('source_base.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def verify_decoupling():

    
    # print(truth_data)
    
    #Const.w_Si = 0.5e-6
    #Const.w0 = 0.7e-6
    Const.eta_norm_coupling = 1 / np.sqrt(Const.w_Si / Const.w0 / np.sqrt(np.pi))
    Const.TM02_phi_coupling = 0
    Const.F_coupling_coeff = Const.eta_norm_coupling * np.exp(1j * Const.TM02_phi_coupling) \
                             * np.sqrt(Const.w_Si / Const.w0 / np.sqrt(np.pi))
    h_neuron = 1e-6
    w_neuron = 0.9e-6
    y0 = 0
    # define neuron number
    neuron_num = 1
    # construct an ONN layer
    onn_layer = ONN_Layer(neuron_number=neuron_num, bound=0, 
                          h_neuron=h_neuron / Const.Lambda0, 
                          w_neuron=w_neuron / Const.Lambda0,
                          y=y0 / Const.Lambda0)
    # define input Ex
    input_Ex = np.ones((1, neuron_num))

    H_area = 10e-6
    sample_num = 2000
    # define dests location
    dests = np.zeros((sample_num, 2))
    
    dests[:, 0] = 1e-9
    dests[:, 1] = np.linspace(y0, -H_area, sample_num)
    Ex = onn_layer.calculate_Ex(input_Ex, dests, verify=True)
    # dests[:, 0] = truth_data[0] * 1e-6 + onn_layer.x0
    
    # Ex, cache = onn_layer.forward_propagation(input_Ex, dests)
    Ex = Ex.reshape((sample_num, ))

    # decoupling_results = np.zeros((sample_num, 2))
    # decoupling_results[:, 0] = dests[:,1]
    # decoupling_results[:, 1] = np.real(Ex)
    # np.savetxt("./results/decoupling_results.csv", decoupling_results, delimiter=',')
    plt.subplot(2, 3, 1)
    truth_data = pd.read_csv('./data/E_radium_w=2.csv', header=None)
    plt.plot(-truth_data[0], truth_data[1], label='simulated')
    plt.plot(dests[:, 1], np.real(Ex), label='modeled')
    plt.xlabel('y position(um)')
    plt.ylabel('Normalized Electrical Field')
    plt.legend()
    plt.xlim([y0, -H_area])
    
    plt.subplot(2, 3, 2)
    truth_data = pd.read_csv('./data/E_abs_radium_w=2.csv', header=None)
    plt.plot(-truth_data[0], truth_data[1], label='simulated')
    plt.plot(dests[:, 1], np.abs(Ex), label='modeled')
    plt.xlabel('y position(um)')
    plt.ylabel('Normalized Amplitude of Electrical Field')
    plt.legend()
    plt.xlim([y0, -H_area])
    # plt.savefig('Gauss-Beam.pdf', format='pdf', bbox_inches='tight')
    
    R0 = 8.5e-6
    theta = np.linspace(-np.pi / 2, np.pi / 2, sample_num)
    dests[:, 0] = R0 * np.sin(theta)
    dests[:, 1] = y0 - h_neuron - R0 * np.cos(theta)
    Ex = onn_layer.calculate_Ex(input_Ex, dests, verify=True)
    Ex = Ex.reshape((sample_num, ))

    plt.subplot(2, 3, 3)
    truth_data = pd.read_csv('./data/E_abs_angle_w=2.csv', header=None)
    plt.plot(-truth_data[0], truth_data[1], label='simulated')
    plt.plot(theta, np.abs(Ex), label='modeled')
    plt.xlabel('Angle(rad)')
    plt.ylabel('Normalized Amplitude of Electrical Field')
    plt.legend()
    plt.xlim([-np.pi / 2, np.pi / 2])

    plt.subplot(2, 3, 4)
    truth_data = pd.read_csv('./data/E_arg_angle_w=2.csv', header=None)
    plt.plot(truth_data[0], truth_data[1], label='simulated')
    plt.plot(theta, arg_cal(Ex), label='modeled')
    plt.xlabel('Angle(rad)')
    plt.ylabel('Phase of Electrical Field')
    plt.legend()
    plt.xlim([-np.pi / 2, np.pi / 2])

    w_chip = 20e-6
    dests[:, 0] = np.linspace(-w_chip / 2, w_chip / 2, sample_num)
    dests[:, 1] = y0 - h_neuron - R0
    Ex = onn_layer.calculate_Ex(input_Ex, dests, verify=True)
    Ex = Ex.reshape((sample_num, ))

    plt.subplot(2, 3, 5)
    truth_data = pd.read_csv('./data/E_abs_horizontal_w=2.csv', header=None)
    plt.plot(truth_data[0], truth_data[1], label='simulated')
    plt.plot(dests[:, 0], np.abs(Ex), label='modeled')
    plt.xlabel('x position (um)')
    plt.ylabel('Normalized Amplitude of Electrical Field')
    plt.legend()
    plt.xlim([-10e-6, 10e-6])    
    # plt.savefig('Gauss-Beam.pdf', format='pdf', bbox_inches='tight')
    
    plt.subplot(2, 3, 6)
    truth_data = pd.read_csv('./data/E_arg_horizontal_w=2.csv', header=None)
    plt.plot(truth_data[0], truth_data[1], label='simulated')
    plt.plot(dests[:, 0], arg_cal(Ex), label='modeled')
    plt.xlabel('x position (um)')
    plt.ylabel('Phase of Electrical Field')
    plt.legend()
    plt.xlim([-10e-6, 10e-6])    
    
    plt.savefig('./figures/Gauss-Beam.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def arg_cal(Ex):
    arg = np.zeros(Ex.shape)
    Ex_real = np.real(Ex)
    Ex_imag = np.imag(Ex)
    Ex_abs = np.abs(Ex)
    s_angel = Ex_imag / Ex_abs
    c_angel = Ex_real / Ex_abs
    arg = np.arccos(c_angel)
    arg[s_angel < 0] = -np.arccos(c_angel)[s_angel < 0]
    return arg

def main():
    test_plot()



if __name__ == '__main__':
    main()
    pass


