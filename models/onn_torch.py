import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import torch
import numpy as np
import utils.helpers
import cmath
import json

from utils import constants as Const
from torch.autograd import Function
class DiffractiveFunction(Function):

    @staticmethod
    def forward(ctx, waves, x_cords, y_cords, x0, y0, h_neuron):
        
        r0 = x_cords - x0
        z = torch.abs(y_cords - (y0 - h_neuron - Const.delta))
        r = torch.sqrt(r0 ** 2 + z ** 2)
        theta = torch.nan_to_num(torch.acos(z / r), nan=0.0)
        # theta = torch.acos(z / r)
        w = Const.w0 * torch.sqrt(1 + (z * Const.Lambda / Const.pi / Const.w0 ** 2) ** 2)
        
        E_RSM_coeff = torch.sqrt(2 * Const.w0 / r / Const.pi_sqrt)
        E_GBM_coeff = torch.exp(-r0 ** 2 / w ** 2) 
        E_RSM = Const.k_RSM * E_RSM_coeff * torch.cos(theta)
        E_GBM = Const.k_GBM * torch.sqrt(Const.w0 / w) * E_GBM_coeff

        F_coupling = Const.F_coupling_coeff
        P_propagation = cmath.exp(-1j * Const.TM02_beta * h_neuron / 2)

        G_decouping = Const.TM02_eta_decoupling * cmath.exp(1j * Const.TM02_phi_decoupling) \
                        * (E_RSM + E_GBM) * torch.exp(-1j * (Const.k_sub * r))
        
        Ex_TEM02 = F_coupling * P_propagation * G_decouping

        dr_dx0 = -1 * r0 / r
        dr0_dx0 = -1

        dE_RSM_dx0 = -1.5 * Const.k_RSM * E_RSM_coeff * z / r ** 2.5 * dr_dx0

        dE_GBM_coeff_dx0 = E_GBM_coeff * (-2) * r0 / w ** 2 * dr0_dx0 
        dE_GBM_dx0 = Const.k_GBM * torch.sqrt(Const.w0 / w) * dE_GBM_coeff_dx0

        dE_dx0 = dE_RSM_dx0 + dE_GBM_dx0
        dG_decouping_dx0 = Const.TM02_eta_decoupling * np.exp(1j * Const.TM02_phi_decoupling) \
                    * ((dE_dx0) *  torch.exp(-1j * (Const.k_sub * r)) + (E_RSM + E_GBM) * \
                    torch.exp(-1j * (Const.k_sub * r)) * -1j * Const.k_sub * dr_dx0)

        dEx_dx0 = F_coupling * P_propagation * dG_decouping_dx0



        ctx.save_for_backward(waves, x0)
class DiffractiveLayer(torch.nn.Module):

    def __init__(self,
                 neuron_number,
                 bound,
                 distance,
                 y,
                 h_neuron=3e-6,
                 w_neuron=0.9e-6,
                 ):
        super(DiffractiveLayer, self).__init__()
        # not use "* lambda0"
        self.neuron_number = neuron_number
        self.h_neuron = h_neuron
        self.w_neuron = w_neuron
        start = bound * Const.Lambda0
        end = bound * Const.Lambda0 + distance * neuron_number * Const.Lambda0
        original_x0 = torch.linspace(start=start, end=end, steps=neuron_number)

        x0 = torch.nn.Parameter(original_x0)
        self.register_parameter("x0", x0)
        # self.register_buffer("h_neuron", torch.Tensor(h_neuron))
        self.register_buffer("original_x0", original_x0)
        self.register_buffer("y0", torch.ones_like(x0) * y * Const.Lambda0)
        # self.register_buffer("w0", torch.Tensor(Const.w0))
        # self.register_buffer("delta", torch.Tensor(Const.delta))
        # self.register_buffer("Lambda". torch.Tensor(Const.Lambda))
        
    def forward(self, waves, x_cords, y_cords):
        """
            x_coords: reshaped from dests[..., 0], shape is (m, l, 1)
            y_coords: reshaped from dests[..., 1], shape is (m, l, 1)
        """

        r0 = x_cords - self.x0
        z = torch.abs(y_cords - (self.y0 - self.h_neuron - Const.delta))
        r = torch.sqrt(r0 ** 2 + z ** 2)
        # theta = torch.nan_to_num(torch.acos(z / r), nan=0.0)
        theta = torch.acos(z / r)
        w = Const.w0 * torch.sqrt(1 + (z * Const.Lambda / Const.pi / Const.w0 ** 2) ** 2)
        
        E_RSM_coeff = torch.sqrt(2 * Const.w0 / r / Const.pi_sqrt)
        E_GBM_coeff = torch.exp(-r0 ** 2 / w ** 2) 
        E_RSM = Const.k_RSM * E_RSM_coeff * torch.cos(theta)
        E_GBM = Const.k_GBM * torch.sqrt(Const.w0 / w) * E_GBM_coeff

        F_coupling = Const.F_coupling_coeff
        P_propagation = cmath.exp(-1j * Const.TM02_beta * self.h_neuron / 2)

        G_decouping = Const.TM02_eta_decoupling * cmath.exp(1j * Const.TM02_phi_decoupling) \
                        * (E_RSM + E_GBM) * torch.exp(-1j * (Const.k_sub * r))
        
        Ex_TEM02 = F_coupling * P_propagation * G_decouping
        Ex = torch.matmul(waves, torch.t(Ex_TEM02))
        return Ex
    
    # def calculate_inside_space(self, waves, x_cords, y_cords):
    #     """
    #         Calculate Inside Field
    #     """
        

    #     # compute whether the x_coords are inside the neurons
    #     Inside = (torch.abs(x_cords - self.x0) < (self.w_neuron / 2))

    #     # print(dests.shape)
    #     # print(self.x0.shape)
    #     # print((x_coords - self.x0).shape)

    #     E_profile_base_model = torch.exp(-(x_cords - self.x0) ** 2 / Const.w0 ** 2)
    #     F_coupling = Const.F_coupling_coeff
    #     P_propagation = torch.exp(-1j * Const.TM02_beta * (y_cords - self.y0))
    #     Ex_TM02 = F_coupling * P_propagation * E_profile_base_model
    #     # construct the inside field model tem02
        
    #     # tem02.inference(x_coords - self.x) returns both Ex and Ey
        
    #     # np.savetxt("Ex_inside.txt", Ex[5,:])
    #     print((Ex_TM02 * Inside).shape)
    #     Ex = torch.matmul(waves, torch.t(Ex_TM02 * Inside))
    #     return Ex


    #     #Ex = Const.Coupling_TM02 * abs(tem02.inference([x_coords - self.x0])[0])*np.exp(1j*(y-y0-h_neuron/2)*(-TEM02.k_eff))*np.exp(1j*TEM02.phi_coupling)
    # def calculate_Ex(self, Ex0, x_cords, y_cords):
    #     """ 
    #     This function caluclates Ex both inside and outside the layer
    #     """

    #     y_bound = self.y0 - self.h_neuron
    #     Ex_inside = self.calculate_inside_space(Ex0, x_cords, y_cords)
    #     Ex = self.verify_forward(Ex0, x_cords, y_cords)
    #     # replace the inside Ex with correct values
    #     mask = y_cords.squeeze(-1) > y_bound
    #     Ex[0, mask] = Ex_inside[0, mask]
    #     # dests[..., 1] < y_bound
    #     return Ex

    # def verify_forward(self, waves, x_cords, y_cords):
    #     """
    #         x_coords: reshaped from dests[..., 0], shape is (m, l, 1)
    #         y_coords: reshaped from dests[..., 1], shape is (m, l, 1)
    #     """
        
    #     r0 = x_cords - self.x0
    #     z = torch.abs(y_cords - (self.y0 - self.h_neuron - Const.delta))
        
    #     w = Const.w0 * torch.sqrt(1 + (z * Const.Lambda / Const.pi / Const.w0 ** 2) ** 2)

    #     F_coupling = Const.F_coupling_coeff
    #     P_propagation = cmath.exp(-1j * Const.TM02_beta * self.h_neuron / 2)

    #     R = z * (1 + (np.pi * Const.w0 ** 2 / z / Const.Lambda) ** 2)
    #     phi = torch.atan(Const.Lambda * z / Const.pi / Const.w0 ** 2)
    #     G_decouping = Const.TM02_eta_decoupling * cmath.exp(1j * Const.TM02_phi_decoupling) \
    #                     * torch.sqrt(Const.w0 / w) * torch.exp(1j * phi / 2) * torch.exp(-r0 ** 2 / w ** 2) \
    #                     * torch.exp(-1j * Const.k_sub * (r0 ** 2 / 2 / R + z) + 1j * Const.TM02_phi_decoupling)
        
    #     Ex_TEM02 = F_coupling * P_propagation * G_decouping
    #     Ex = torch.matmul(waves, torch.t(Ex_TEM02))
    #     return Ex

class DiffractiveNetwork(torch.nn.Module):

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
                compact_decoding=False,):
        
        super(DiffractiveNetwork, self).__init__()

        self.input_distance = input_distance
        self.input_bound = input_bound

        self.hidden_distance = hidden_distance
        self.hidden_bound = hidden_bound

        self.output_distance = output_distance
        self.output_bound = output_bound

        self.input_neuron_num = input_neuron_num
        self.hidden_neuron_num = hidden_neuron_num
        self.output_neuron_num = output_neuron_num

        self.layer_num = hidden_layer_num + 1

        # define layers parameters
        neuron_nums = [input_neuron_num] + hidden_neuron_num
        bounds = [input_bound] + hidden_bound
        distances = [input_distance] + hidden_distance
        y = [0]
        for i in range(hidden_layer_num):
            y.append(y[-1] + hidden_layer_distance[i])

        self.layers = torch.nn.ModuleList([DiffractiveLayer(neuron_nums[i], bounds[i], distances[i], y[i]) for i in range(self.layer_num)])
        
        # define output layer
        start = output_bound * Const.Lambda0
        end = (output_bound + output_distance * output_neuron_num) * Const.Lambda0
        x_out = torch.linspace(start, end, output_neuron_num)
        self.register_buffer("x_out", x_out)
        self.register_buffer("y_out", torch.ones_like(x_out) * (y[-1] + output_layer_distance) * Const.Lambda0)

        # define softmax layer
        self.m = torch.nn.Softmax(dim=-1)

    def forward(self, waves):
        for index, layer in enumerate(self.layers[:-1]):
            x_cords = self.layers[index + 1].x0.unsqueeze(-1)
            y_cords = self.layers[index + 1].y0.unsqueeze(-1)
            waves = layer.forward(waves, x_cords, y_cords)
        
        waves = self.layers[-1].forward(waves, self.x_out.unsqueeze(-1), self.y_out.unsqueeze(-1))
        waves_abs = torch.abs(waves)
        
        return self.m(waves_abs)
    
def main():
    x = torch.randn(4, 3, 2)
    y = x[..., 0:1]
    z = x[..., 1:2]

    n = 40
    d = 6

    bound = utils.helpers.get_bound(n, d, 250)
    layer = DiffractiveLayer(neuron_number=n, bound=bound * Const.Lambda0, distance=d * Const.Lambda0, y=0,)  

    on = 10
    d = 20
    ld = 200e-6
    bound = utils.helpers.get_bound(n, d, 250)
    x_cords = torch.arange(bound * Const.Lambda0, bound * Const.Lambda0 + d * Const.Lambda0 * on, d * Const.Lambda0).unsqueeze(-1)
    y_cords = torch.ones(on).unsqueeze(-1) * ld

    waves = torch.rand(10, n) * (1+0j)
    output_waves = layer.forward(waves, x_cords, y_cords)
    # print(output_waves)

    # read json
    json_file = './json/example.json'
    with open(json_file, 'r') as f:
        DONNs = json.load(f)["ONN"]
    # parse json

    input_dim = 100
    output_dim = 10
    for DONN_index, DONN in enumerate(DONNs):
        hidden_neuron_num_list = []
        hidden_neuron_distance_list = []
        hidden_bound_list = []
        hidden_layer_distance_list = []
    
        print("---------DONN No. %d---------" % DONN_index)
        input_neuron_distance = DONN["input_neuron_distance"]
        
        max_width = (input_dim + 10) * input_neuron_distance
        hidden_layers = DONN["hidden_layers"]

        for layer_index, layer in enumerate(hidden_layers):
            hidden_neuron_num_list.append(layer["neuron_number"])
            hidden_neuron_distance_list.append(layer["neuron_distance"])
            hidden_layer_distance_list.append(layer["layer_distance"])
            max_width = max((layer["neuron_number"] + 10) * layer["neuron_distance"], max_width)
        
        output_neuron_distance = DONN["output_neuron_distance"]
        max_width = max((output_dim + 10) * output_neuron_distance, max_width)

        input_bound = utils.helpers.get_bound(input_dim, input_neuron_distance, max_width)
        for layer_index, layer in enumerate(hidden_layers):
            hidden_bound = utils.helpers.get_bound(layer["neuron_number"], layer["neuron_distance"], max_width)
            hidden_bound_list.append(hidden_bound)
        output_bound = utils.helpers.get_bound(output_dim, output_neuron_distance, max_width)

        donn_model = DiffractiveNetwork(input_neuron_num=input_dim,
                        input_distance=input_neuron_distance,
                        input_bound=input_bound,
                        output_neuron_num=output_dim,
                        output_distance=output_neuron_distance,
                        output_bound=output_bound,
                        hidden_layer_num=len(hidden_layers),
                        hidden_neuron_num=hidden_neuron_num_list,
                        hidden_distance=hidden_neuron_distance_list,
                        hidden_bound=hidden_bound_list,
                        hidden_layer_distance=hidden_layer_distance_list,
                        output_layer_distance=DONN["output_layer_distance"],
                        phi_init="default",
                        nonlinear=False,
                        compact_decoding=False,
        )

        waves = torch.ones(1, input_dim) * (1+0j)
        waves = donn_model.forward(waves)




if __name__ == "__main__":
    main()
        

        
