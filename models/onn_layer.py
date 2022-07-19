from __future__ import absolute_import, division, print_function, unicode_literals
from cmath import cos
import sys, os
from traceback import print_tb
from matplotlib.pyplot import axis

from torch import batch_norm, zeros
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
from utils import constants as Const
from models.tem02 import TEM02

class ONN_Layer(object):
    """
    The Layer of ONN 
    
    Attributes:
        neuron_number: neuron number
        distance: distance between neighbour neurons
        bound: boundary distance between the first neuron and the left limit,
               also the distance between the last and the right limit
        x, y: coordinates for neurons
        Ex_0: input Ex of the neurons
        input coords: input coords of the neurons, shape is (n, 2) or (m, n, 2)

    """
    def __init__(self,
                 neuron_number=40,
                 distance=6,
                 y=0,
                 bound=10,
                 h_neuron=2,
                 w_neuron=0.6,
                 phi_init="default",
                 ) -> None:

        super().__init__()
        self.neuron_number = neuron_number
        self.distance = distance * Const.Lambda0
        self.bound = bound * Const.Lambda0

        # initiate phase change
        if phi_init == "default":
            self.phi = np.zeros(neuron_number)
        elif phi_init == "random":
            self.phi = np.random.rand(neuron_number) * np.pi * 2

        self.h_neuron = np.ones(neuron_number, dtype=np.float64) * h_neuron * Const.Lambda0 # h_neuron is an numpy array
        self.length = (bound * 2 + distance * (neuron_number - 1)) * Const.Lambda0 # the total length of the layer
        self.w_neuron = w_neuron * Const.Lambda0
        # define batch layer and input Ex
        # Ex0 shape: (m, neuron_number)

        # x0_bound is the half bound of the cloest neuron
        self.x0_bound = Const.Lambda0 * distance / 2

        # the x coordinates for the input layer is an numpy array

        # self.coords = np.zeros([neuron_number, 2], dtype=np.float64)
        self.x0 = np.arange(bound, bound + distance * neuron_number, distance) * Const.Lambda0
        self.x0_left_limit = self.x0 - Const.Lambda0 * Const.x0_bound
        self.x0_right_limit = self.x0 + Const.Lambda0 * Const.x0_bound

        self.original_x0 = np.copy(self.x0)
        self.original_phi = np.copy(self.phi)
        # self.coords[:, 0] = np.arange(bound, bound + distance * neuron_number, distance) * Const.Lambda0
        # the default y coordinate for the layer is 0
        self.y0 = y * Const.Lambda0
        # self.coords[:, 1] = np.full(self.neuron_number, fill_value=y * Const.Lambda0)
        """
        # input coords
        # y = y0 + h_neuron / 2
        self.input_coords = np.zeros([neuron_number, 2], dtype=np.float64)
        self.input_coords[:, 0] = self.coords[:, 0]
        self.input_coords[:, 1] = self.coords[:, 1] + self.h_neuron / 2
        """
        
    def calculate_r(self, x: np.array, y: np.array) -> np.array:
        """
        self.x0: [n0, n1, n2, ...]
                 dimension: [neuron_num]
        x, y: [[x0], [x1], [x2], ...]
              [[y0], [y1], [y2], ...]
              dimension: [..., 1]
        x - self.x0: [[x0-n0, x0-n1, x0-n2, ...], [x1-n0, x1-n1, x1-n2, ...], ...]
                     dimension: [..., 40]
        """
        z = self.calculate_z(x, y)
        r = np.sqrt((x - self.x0) ** 2 + z ** 2)
        return r
    
    def calculate_z(self, x: np.array, y: np.array) -> np.array:
        z = np.abs(y - (self.y0 - self.h_neuron - Const.delta))
        return z

    def calculate_theta(self, x: np.array, y: np.array) -> np.array:
        z = self.calculate_z(x, y)
        r = self.calculate_r(x, y)
        theta = np.arccos(z / r)
        theta[np.isnan(theta)] = 0
        return theta

    def convert_dests(self, dests: np.array):
        """
            Convert dests to x and y coordinates as required
        """
        output_dim = dests.shape[:-1]
        x_coords = dests[..., 0]
        y_coords = dests[..., 1]
        x_coords = x_coords.reshape(output_dim + (1, ))
        y_coords = y_coords.reshape(output_dim + (1, ))
        return (x_coords, y_coords)

    
    def input_Ex(self, x: np.array, y: np.array) -> np.array:
        """
            Define the base mode Ex of input light source (-12.5 um to 12.5um)
        """
        r0 = x - self.x0
        z = np.abs(y - self.y0)
        Ex = np.cos(r0 / 8e-6) * np.exp(1j * z * Const.k_sub)
        return Ex

    def calculate_inside_space(self, dests: np.array):
        """
            Calculate Inside Field
        """
        output_dim = dests.shape[:-1]
        
        # the dests plane should below the incident plane
        # assert self.coords[:,1] - self.h_neuron / 2 > dests[:,1]
        
        # x and y coordinates
        x_coords = dests[..., 0]
        y_coords = dests[..., 1]
        x_coords = x_coords.reshape(output_dim + (1, ))
        y_coords = y_coords.reshape(output_dim + (1, ))

        # compute whether the x_coords are inside the neurons
        Inside = (np.abs(x_coords - self.x0) < (self.w_neuron / 2))

        # print(dests.shape)
        # print(self.x0.shape)
        # print((x_coords - self.x0).shape)

        E_profile_base_model = np.exp(-(x_coords - self.x0) ** 2 / Const.w0 ** 2)
        F_coupling = Const.F_coupling_coeff
        P_propagation = np.exp(-1j * Const.TM02_beta * (y_coords - self.y0))
        Ex = F_coupling * P_propagation * E_profile_base_model
        # construct the inside field model tem02
        
        # tem02.inference(x_coords - self.x) returns both Ex and Ey
        
        Ex = Ex * Inside
        # np.savetxt("Ex_inside.txt", Ex[5,:])
        return Ex


        #Ex = Const.Coupling_TM02 * abs(tem02.inference([x_coords - self.x0])[0])*np.exp(1j*(y-y0-h_neuron/2)*(-TEM02.k_eff))*np.exp(1j*TEM02.phi_coupling)

    def calculate_outside_space(self, dests, verify=False):
        """ 
            Calculate Outside Field

        Input:
            Ex0: input Ex0 of shape (k, n) where k is the input image number and n is the 
                 input neuron number
            dests: an numpy array defines the coordinates of the output, shape is (m, l, 2),
                   currently, it only support N-D array. For example, a 3-D array:
            [
                [[x0, y], [x1, y], [x2, y], ...], -> each corresponding to one graph
                [[x0, y], [x1, y], [x2, y], ...],
                [[x0, y], [x1, y], [x2, y], ...],
                ...
            ]

        Variables:
            x_coords: reshaped from dests[..., 0], shape is (m, l, 1)
            y_coords: reshaped from dests[..., 1], shape is (m, l, 1)
            Ex_indie: independent Ex of each neuron on the destination
            [
                [
                [Ex[n0->x0], Ex[n1->x0], Ex[n2->x0], ...],
                [Ex[n0->x1], Ex[n1->x1], Ex[n2->x1], ...],
                [Ex[n0->x2], Ex[n1->x2], Ex[n2->x2], ...],
                ...
                ]
                ...
            ]
        
        Output:
            Ex: accumulation of independent Ex
            [
                [Ex_x0, Ex_x1, Ex_x2, ... ]
            ]
            shape is (m, l)
        """

        output_dim = dests.shape[:-1]
        
        # the dests plane should below the incident plane
        # assert self.coords[:,1] - self.h_neuron / 2 > dests[:,1]
        # x and y coordinates
        cache = {}

        x_coords = dests[..., 0]
        y_coords = dests[..., 1]
        x_coords = x_coords.reshape(output_dim + (1, ))
        y_coords = y_coords.reshape(output_dim + (1, ))

        r = self.calculate_r(x_coords, y_coords)
        z = self.calculate_z(x_coords, y_coords)
        # print("r shape: ", r.shape)
        # print("z shape: ", z.shape)

        theta = self.calculate_theta(x_coords, y_coords)
        # what does w0 mean?
        # now define w0 as global constant
        w0 = Const.w0

        #A = 4e7 * 2.5 / (1.1e-4 + z ** 0.7) / 17000
        #A = 1e4 * 2.5 / (1.1e-4 + z ** 0.7) / 17000
        # R = z * (1 + (np.pi * w0 ** 2 / z / Const.Lambda ) ** 2)
        # phi = np.arctan(Const.Lambda * z / np.pi / w0 ** 2) * 1

        r0 = x_coords - self.x0
        w = w0 * np.sqrt(1 + (z * Const.Lambda / np.pi / w0 ** 2) ** 2)
        
        E_RSM_coeff = np.sqrt(2 * w0 / r / np.sqrt(np.pi))
        E_GBM_coeff = np.exp(-r0 ** 2 / w ** 2)
        E_RSM = Const.k_RSM * E_RSM_coeff * np.cos(theta)
        E_GBM = Const.k_GBM * np.sqrt(w0 / w) * E_GBM_coeff

        F_coupling = Const.F_coupling_coeff
        P_propagation = np.exp(-1j * Const.TM02_beta * self.h_neuron / 2)
        

        if verify:
            r0 = x_coords - self.x0
            R = z * (1 + (np.pi * w0 ** 2 / z / Const.Lambda) ** 2)
            phi = np.arctan(Const.Lambda * z / np.pi / w0 ** 2)
            G_decouping = Const.TM02_eta_decoupling * np.exp(1j * Const.TM02_phi_decoupling) \
                          * np.sqrt(w0 / w) * np.exp(1j * phi / 2) * np.exp(-r0 ** 2 / w ** 2) \
                          * np.exp(-1j * Const.k_sub * (r0 ** 2 / 2 / R + z) + 1j * Const.TM02_phi_decoupling)
        # accutully used in calculation
        else: 
            G_decouping = Const.TM02_eta_decoupling * np.exp(1j * Const.TM02_phi_decoupling) \
                        * (E_RSM + E_GBM) * np.exp(-1j * (Const.k_sub * r))
            
            # calculate the derivatives of the dG_decouping_dx0
            dr_dx0 = -1 * r0 / r
            dr0_dx0 = -1
            """
            the following gradient calculation is angle_dependent (do not use)
            
            
            # dtheta_dx0 = -1 / np.sqrt(1 - (z / r) ** 2) * (-dr_dx0 * z) / r ** 2
            # repair those dtheta_dx0 of z == r using approximate theory
            # when z = 0, dcos(theta) = -dx / r
            # dtheta_dx0[np.isnan(dtheta_dx0)] = -1 / r[np.isnan(dtheta_dx0)]
            # dE_RSM_coeff_dx0 = 1 / 2 / E_RSM_coeff * (-dr_dx0 * 2 * w0 / np.sqrt(np.pi) / r ** 2) * np.cos(theta)
            # dcos_dx0 = E_RSM_coeff * (-np.sin(theta)) * dtheta_dx0
            # dE_RSM_dx0 = Const.k_RSM * (dE_RSM_coeff_dx0 * np.cos(theta) + E_RSM_coeff * dcos_dx0)
            
            # the following gradient calculation neglects cos(theta)

            """

            dE_RSM_dx0 = -1.5 * Const.k_RSM * E_RSM_coeff * z / r ** 2.5 * dr_dx0

            dE_GBM_coeff_dx0 = E_GBM_coeff * (-2) * r0 / w ** 2 * dr0_dx0 
            dE_GBM_dx0 = Const.k_GBM * np.sqrt(w0 / w) * dE_GBM_coeff_dx0

            dE_dx0 = dE_RSM_dx0 + dE_GBM_dx0
            dG_decouping_dx0 = Const.TM02_eta_decoupling * np.exp(1j * Const.TM02_phi_decoupling) \
                        * ((dE_dx0) *  np.exp(-1j * (Const.k_sub * r)) + (E_RSM + E_GBM) *  np.exp(-1j * (Const.k_sub * r)) * -1j * Const.k_sub * dr_dx0)

            cache["dEx_dx0"] = F_coupling * P_propagation * dG_decouping_dx0
        
        Ex = F_coupling * P_propagation * G_decouping

        return Ex, cache

    def forward_propagation(self, Ex0, dests):
        """
            Forward Propagation
        """

        method = 2


        # TEM02_w = Const.Coupling_TM02 * Ex_TEM02
        Ex_TEM02, cache = self.calculate_outside_space(dests)
        Ex_TEM02 = Ex_TEM02 * np.exp(1j * self.phi)

        # print("Ex_TEM02 shape: ", Ex_TEM02.shape)
        # print("h_w shape: ", h_w.shape)
        # print("Ex_TM02 shape: ", Ex_w.shape)
        # print("Ex_0 shape: ", Ex0.shape)

        if method == 1:
            # method 1
            # expand Ex0 dimensions
            dim = Ex_TEM02.ndim
            for i in range(dim - 1):
                Ex_new = np.expand_dims(Ex0, axis=1)

            Ex_TM02 = Ex_new * Ex_TEM02
            Ex_indie = Ex_TM02
            Ex = np.sum(Ex_indie, axis=-1)
        
        else:
            # method 2 (currently used for backward propagation)
            # y = x * W
            Ex_w = Ex_TEM02.T
            Ex = Ex0.dot(Ex_w)
            # cache = {}
            cache["Ex_TEM02"] = Ex_TEM02
            cache["Ex0"] = Ex0
            cache["Ex_w"] = Ex_w
            cache["TEM02_w"] = Ex_TEM02
            cache["h_w"] = Ex_TEM02

        return Ex, cache

    def calculate_Ex(self, Ex0, dests, verify=False):
        """ 
        This function caluclates Ex both inside and outside the layer
        """

        y_bound = self.y0 - self.h_neuron[0]
        Ex_inside = self.calculate_inside_space(dests)
        Ex, _ = self.calculate_outside_space(dests, verify)
        # replace the inside Ex with correct values
        Ex[dests[..., 1] > y_bound] = Ex_inside[dests[..., 1] > y_bound]
        # dests[..., 1] < y_bound
        return Ex0.dot(Ex.T)



    def backward_propagation(self, dout_r, dout_i, cache, y=None):
        """
        Compute the Loss and Gradient of Backward Propagation

        Input:
            Ex0: input Ex0 of shape (k, n) where k is the input image number and n is the 
                 input neuron number
            dests: an numpy array defines the coordinates of the output, shape is (m, l, 2),
                   currently, it only support N-D array. For example, a 3-D array:
            [
                [[x0, y], [x1, y], [x2, y], ...], -> each corresponding to one graph
                [[x0, y], [x1, y], [x2, y], ...],
                [[x0, y], [x1, y], [x2, y], ...],
                ...
            ]
            y: ground truth values for dests
        
        Output:
            loss: 
        """
        Ex_TEM02 = cache["Ex_TEM02"]
        Ex0 = cache["Ex0"]
        Ex_w = cache["Ex_w"]
        TEM02_w = cache["TEM02_w"]
        h_w = cache["h_w"]
        # print("Ex0 shape: ", Ex0.shape)
        # print("Ex_w shape: ", Ex_w.shape)
        """
            Ry = Rx * Rw - Ix * Iw
            Iy = Rx * Iw + Ix * Rw
            
            dRy_dRx = dRy * Rw.T
            dRy_dIx = dRy * Iw.T
            dIy_dRx = dIy * -Iw.T
            dIy_dIx = dIy * Rw.T

            dRy_dRw = Rx.T * dRy
            dRy_dIw = -Ix.T * dRy
            dIy_dRw = Ix.T * dIy
            dIy_dIw = Rx.T * dIy

        """
        # 
        dx_r = dout_r.dot(np.real(Ex_w.T)) - dout_i.dot(np.imag(Ex_w.T))
        dx_i = dout_r.dot(np.imag(Ex_w.T)) + dout_i.dot(np.real(Ex_w.T))
        # print("dx_r shape: ", dx_r.shape)
        # print("dx_i shape: ", dx_i.shape)

        dw_r = np.real(Ex0.T).dot(dout_r) + np.imag(Ex0.T).dot(dout_i)
        dw_i = -np.imag(Ex0.T).dot(dout_r) + np.real(Ex0.T).dot(dout_i)

        """
            Rw = Rh * Rtem02 - Ih * Item02
            Iw = Rh * Item02 + Ih * Rtem02
        """
        angle = Const.TM02_k_eff * self.h_neuron + Const.TM02_phi_coupling

        dh_r = - dw_r.T * np.real(Ex_TEM02) * np.sin(angle) * Const.TM02_k_eff - np.imag(Ex_TEM02) * np.cos(angle) * Const.TM02_k_eff
        dh_i = dw_i.T * np.real(Ex_TEM02) * np.cos(angle) * Const.TM02_k_eff - np.imag(Ex_TEM02) * np.sin(angle) * Const.TM02_k_eff
        dh = np.sum(dh_r + dh_i, axis=0)
        # print("dw_r shape: ", dw_r.shape)
        # print("TEM02_w shape: ", TEM02_w.shape)
        # print("dh shape: ", dh.shape)

        return dx_r, dx_i, dh


    def backward_propagation_holomorphic_phi(self, dout, cache, y=None):
        """
        Compute Backward Propagation with Holomorphic Assumption
        with respect to phi


        Output_Ex = Input_Ex * Ex_w
        (n, m_out)  (n, m_in)  (m_in, m_out)

        Ex_w = TEM02 * h_w * exp(phi * 1j) (Transpose)
        dphi = dw * Ex_w * 1j
                (m_in, m_out)
        Input:
            dout: shape of (n, m_out)
            cache: caches used for backward propagation
        Output:
            dphi: 
            dx: 
        """
        
        Ex0 = cache["Ex0"]
        Ex_w = cache["Ex_w"]

        # print("Ex0 shape: ", Ex0.shape)
        # print("Ex_w shape: ", Ex_w.shape)
        dx = dout.dot(Ex_w.T)
        dw = Ex0.T.dot(dout)
        dphi = dw * Ex_w * 1j
        
        return dx, dphi.T

    def backward_propagation_holomorphic_x0(self, dout, cache, y=None):
        """
        Compute Backward Propagation with Holomorphic Assumption
        with respect to x0

        Output_Ex = Input_Ex * Ex_w
        (n, m_out)  (n, m_in)  (m_in, m_out)

        Ex_w = TEM02 = F_coupling * P_propagation * G_decouping(x0)
        dEx_w = dw * Ex_w * dG_decouping / dx0

        """

        # Shape: 
        # Ex0 (n, m_in) 
        # Ex_w (m_in, m_out)
        # dout (n, m_out)
        # dEx_dx0 (m_out, m_in) 
        # dx0 (n, m_in, m_out)
        # dx (n, m_in)

        Ex0 = cache["Ex0"]
        Ex_w = cache["Ex_w"]

        # print("Ex_w shape:", Ex_w.shape)
        dEx_dx0 = cache["dEx_dx0"]
        # print("dEx_dx0 shape:", dEx_dx0.shape)
        # print("dEx_dx0:", dEx_dx0)
        dx = dout.dot(Ex_w.T)
        # dw = Ex0.T.dot(dout).T
        # print("Ex0 shape:", Ex0.shape)
        # print("dout shape:", dout.shape)
        
        Ex0 = np.expand_dims(Ex0, axis=-1)
        dout = np.expand_dims(dout, axis=-2)
        dx0 = Ex0 * dout * dEx_dx0.T
        # print("Ex0 shape:", Ex0.shape)
        # print("dout shape:", dout.shape)
        # dw = Ex0.T.dot(dout)
        # print("dw shape:", dw.shape)
        # dx0 = dw.T * dEx_dx0
        # print("dx0 shape:", dx0.shape)
        return dx, dx0    


    def backward_propagation_holomorphic_x0_v2(self, dout, cache, Ex_conj):
        """
        Compute Backward Propagation with Holomorphic Assumption
        with respect to x0

        Output_Ex = Input_Ex * Ex_w
        (n, m_out)  (n, m_in)  (m_in, m_out)

        Ex_w = TEM02 = F_coupling * P_propagation * G_decouping(x0)
        dEx_w = dw * Ex_w * dG_decouping / dx0

        """

        # Shape: 
        # Ex0 (n, m_in) 
        # Ex_w (m_in, m_out)
        # dout (n, m_out)
        # dEx_dx0 (m_out, m_in) 
        # dx0 (n, m_in, m_out)
        # dx (n, m_in)

        Ex0 = cache["Ex0"]
        Ex_w = cache["Ex_w"]
        dEx_dx0 = cache["dEx_dx0"]
        dx = dout.dot(Ex_w.T)
        dx0 = (dout * Ex_conj).dot(dEx_dx0) * Ex0
        return dx, dx0 

def verify():
    L_area = 25 * Const.Lambda0
    x = np.linspace()
        
def main():
    """
    Test ONN Layer
    """
    a = np.array([1 ,2, 3, 4]).reshape((2,2,1))
    b = np.array([1 ,2, 3])
    d = a - b
    e = np.sum(d, axis=-1)
    #print(a)
    #print(d)
    #print(e)


    a = np.array([1,2,3]).reshape((3,1))
    b = a.reshape((1,3))
    d = a - b
    #print(a)
    #print(b)
    #print(d)


    neuron_number = 40
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

    output_dim = dests.shape[-2]
    x = dests[:, 0]
    y = dests[:, 1]
    x = x.reshape(x.shape[:-1] + (output_dim, 1))
    y = y.reshape(y.shape[:-1] + (output_dim, 1))

    image_number = 4
    Ex0 = np.ones((image_number, neuron_number))

    Ex, cache = test_layer.forward_propagation(Ex0, dests)
    Ex = np.abs(Ex)

    print(Ex)

    dout = [np.expand_dims(np.ones(hidden_num), axis=1)] * test_layer.output_neuron_num
    S = np.zeros(image_number + (test_layer.neuron_number, ))



if __name__ == '__main__':
    main()
