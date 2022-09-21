import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import json

import tensorflow as tf
import numpy as np

import utils.helpers
from utils import constants as Const

class DiffractiveLayer(tf.keras.layers.Layer):

    def __init__(self,
                 neuron_number=40,
                 distance=6,
                 y=0,
                 y_next=None,
                 bound=10,
                 h_neuron=2,
                 w_neuron=0.6,
                 mode="x0",
                 ):
        
        super(DiffractiveLayer, self).__init__()

        self.neuron_number = neuron_number
        self.distance = distance * Const.Lambda0
        self.bound = bound * Const.Lambda0

        self.h_neuron = tf.constant(h_neuron * Const.Lambda0, dtype=tf.float32) # h_neuron is an numpy array
        self.length = (bound * 2 + distance * (neuron_number - 1)) * Const.Lambda0 # the total length of the layer
        self.w_neuron = w_neuron * Const.Lambda0
        self.x0_bound = Const.Lambda0 * distance / 2

        x0 = np.linspace(self.bound, self.bound + self.distance * (self.neuron_number - 1), self.neuron_number)

        # initiate phase change
        if mode == "input":
            self.phase = self.add_weight(name="phase", shape=[1, self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
            self.x0 = self.add_weight(name="x0", shape=[self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(x0), trainable=False)
        elif mode == "x0":
            self.phase = self.add_weight(name="phase", shape=[1, self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
            self.x0 = self.add_weight(name="x0", shape=[self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(x0), trainable=True)
        elif mode == "phase":
            self.phase = self.add_weight(name="phase", shape=[1, self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(0.5), trainable=True)
            self.x0 = self.add_weight(name="x0", shape=[self.neuron_number], dtype=tf.float32, initializer=tf.constant_initializer(x0), trainable=False)

        # self.x0_left_limit = x0 - Const.Lambda0 * Const.x0_bound
        # self.x0_right_limit = x0 + Const.Lambda0 * Const.x0_bound

        self.original_x0 = np.copy(self.x0)
        self.y0 = y * Const.Lambda0

        dy = (y_next - y) * Const.Lambda0
        self.z = tf.constant(tf.abs(dy + self.h_neuron + Const.delta), dtype=tf.float32)
        
    def call(self, x, waves):

        w0 = tf.constant(Const.w0, dtype=tf.float32)
        pi_sqrt = tf.constant(np.sqrt(np.pi), dtype=tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        
        xg1 = tf.expand_dims(self.x0, axis=-1)
        dx = x - xg1
        r = tf.sqrt((dx) ** 2 + self.z ** 2)

        w = Const.w0 * tf.sqrt(1 + (self.z * Const.Lambda / pi / Const.w0 ** 2) ** 2)
        
        E_RSM_coeff = tf.sqrt(2 * w0 / r / pi_sqrt)

        E_RSM = Const.k_RSM * E_RSM_coeff * self.z / r

        E_GBM_coeff = tf.exp(-dx ** 2 / w ** 2)

        E_GBM = Const.k_GBM * tf.sqrt(w0 / w) * E_GBM_coeff
        
        F_coupling = tf.constant(Const.F_coupling_coeff, dtype=tf.complex64)

        phase = tf.constant(Const.TM02_beta, dtype=tf.float32) * self.h_neuron / 2
        P_propagation = tf.complex(tf.cos(-phase), tf.sin(-phase))

        TM02_phi_decoupling = tf.constant(Const.TM02_phi_decoupling, dtype=tf.float32)
        TM02_eta_decoupling = tf.constant(Const.TM02_eta_decoupling, dtype=tf.complex64)
        k_sub = tf.constant(Const.k_sub, dtype=tf.float32)
        G_decouping = TM02_eta_decoupling * tf.complex(tf.cos(TM02_phi_decoupling), tf.sin(TM02_phi_decoupling)) \
                      * (tf.cast(E_RSM, dtype=tf.complex64) + tf.cast(E_GBM, dtype=tf.complex64)) * tf.complex(tf.cos(-k_sub * r), tf.sin(-k_sub * r))

        Ex = F_coupling * P_propagation * G_decouping
        
        # propagtion
        theta = pi * self.phase
        y = tf.multiply(waves, tf.complex(tf.cos(theta), tf.sin(theta)))
        y = tf.matmul(y, Ex)
        
        return y

class DONN(tf.keras.Model):

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
                output_dim):

        super(DONN, self).__init__()   
        self.dls = [] # diffractive layers
        self.layer_num = hidden_layer_num + 1
        
        # define layer distance
        self.ys = [0]
        for i in range(hidden_layer_num):
            self.ys.append(self.ys[-1] + hidden_layer_distance[i])
        self.ys.append(self.ys[-1] + output_layer_distance)

        # input layer
        self.dls.append(DiffractiveLayer(neuron_number=input_neuron_num, 
                                            distance=input_distance,
                                            bound=input_bound,
                                            y=self.ys[0], 
                                            y_next=self.ys[1],
                                            mode="input"))
        # hidden layer
        for i in range(hidden_layer_num):
            
            self.dls.append(DiffractiveLayer(neuron_number=hidden_neuron_num[i],
                                                distance=hidden_distance[i],
                                                bound=hidden_bound[i], 
                                                y=self.ys[i + 1], 
                                                y_next=self.ys[i + 2],
                                                mode="x0"))
        
        # detector and output layer
        x0_d = np.linspace(output_bound, output_bound + output_distance * (output_neuron_num - 1), output_neuron_num) * Const.Lambda0 

        dw = np.zeros([output_neuron_num, output_dim]) # detector to output weight
        dn = output_neuron_num // (output_dim * 2 + 1)
        rn = (output_neuron_num % (output_dim * 2 + 1)) // 2 - 1

        for i in range(output_dim):
            dw[rn+ dn * (2 * i + 1) : rn + dn * (2 * i + 2), i] = 255
        self.x0_d = self.add_weight(name="x0_d", shape=[output_neuron_num], dtype=tf.float32, initializer=tf.constant_initializer(x0_d), trainable=False)
        self.dw = tf.constant(dw, dtype=tf.float32)

        # detector and output layer with 10 detectors only
        x0_d = np.linspace(output_bound, output_bound + output_distance * (output_neuron_num - 1), output_neuron_num) * Const.Lambda0 
        dw = np.zeros([output_neuron_num, output_dim]) # detector to output weight
        x0_n = np.zeros([output_dim])
        # dn = output_neuron_num // (output_dim * 2 + 1)

        dn = output_neuron_num // (output_dim)
        if (dn % 2) == 0:
            dn -= 1
        rn = (output_neuron_num - dn * output_dim) // 2 - 1
        for i in range(output_dim):
            x0_n[i] = x0_d[rn + i * dn + (dn + 1) // 2]
        self.x0_d = self.add_weight(name="x0_d", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(x0_n), trainable=False)
        
        

    def call(self, x):
        for i in range(self.layer_num - 1):
            x = self.dls[i](self.layers[i + 1].x0, x)
        x = self.dls[self.layer_num - 1](self.x0_d, x)
        x = tf.abs(x)
        # x = tf.matmul(x, self.dw) 1
        x = tf.keras.activations.softmax(x)
        return x

def test_DONN():

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', dest='size', default=14, type=int, help='dimension of the input image')
    parser.add_argument('--mode', dest='mode', default='x0', type=str, help='mode of the configuration')
    parser.add_argument('--encoding', dest='encoding', default='amplitude', type=str, help='encoding of the input signal')

    parser.add_argument('--json_file', dest='json_file', default='./json/example.json', type=str, help='structure definition json file')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.5e-11, type=float, help='learning rate of the DONN')
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int, help='the number of epochs to train the model')
    parser.add_argument('--batch_size', dest='batch_size', default=50, type=int, help='batch size of the model')
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='print=')
    parser.add_argument('--constrained', dest='constrained', default=False, type=bool, help='with constrained neurons location')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.95, type=float, help='learning rate decay')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='temp', type=str, help='checkpoint_name')

    parser.add_argument('--not_plot', action='store_true', dest='not_plot', help='do not plot the structure of DONN')
    parser.add_argument('--assessment', action='store_true', dest='assessment', help='assess the DONN structure')
    parser.add_argument('--num_assess', dest='num_assess', default=100, type=int, help='number of iterations during struture assessment')
    
    # double decoding
    parser.add_argument('--compact_decoding', action='store_true', dest='compact_decoding', help='decoding in a compact way')
    parser.add_argument('--output_dim', dest='output_dim', default=10, type=int, help='dimension of the output decoder')
    args = parser.parse_args()


    input_dim = args.size ** 2
    output_dim = args.output_dim

    # read json
    with open(args.json_file, 'r') as f:
        DONNs = json.load(f)["ONN"]
    # parse json

    for DONN_index, donn in enumerate(DONNs):
        hidden_neuron_num_list = []
        hidden_neuron_distance_list = []
        hidden_bound_list = []
        hidden_layer_distance_list = []
    
        print("---------DONN No. %d---------" % DONN_index)
        input_neuron_distance = donn["input_neuron_distance"]
        
        max_width = (input_dim + 10) * input_neuron_distance
        hidden_layers = donn["hidden_layers"]

        for layer_index, layer in enumerate(hidden_layers):
            hidden_neuron_num_list.append(layer["neuron_number"])
            hidden_neuron_distance_list.append(layer["neuron_distance"])
            hidden_layer_distance_list.append(layer["layer_distance"])
            max_width = max((layer["neuron_number"] + 10) * layer["neuron_distance"], max_width)
        
        output_neuron_distance = donn["output_neuron_distance"]
        max_width = max((donn["output_neuron_number"] + 10) * output_neuron_distance, max_width)

        input_bound = utils.helpers.get_bound(input_dim, input_neuron_distance, max_width)
        for layer_index, layer in enumerate(hidden_layers):
            hidden_bound = utils.helpers.get_bound(layer["neuron_number"], layer["neuron_distance"], max_width)
            hidden_bound_list.append(hidden_bound)
        output_bound = utils.helpers.get_bound(donn["output_neuron_number"], output_neuron_distance, max_width)

        with tf.GradientTape(persistent=True) as tape:
            waves = tf.ones([10, input_dim], dtype=tf.complex64)
            tape.watch(waves)
            net = DONN(input_neuron_num=input_dim,
                        input_distance=input_neuron_distance,
                        input_bound=input_bound,
                        output_neuron_num=donn["output_neuron_number"],
                        output_distance=output_neuron_distance,
                        output_bound=output_bound,
                        hidden_layer_num=len(hidden_layers),
                        hidden_neuron_num=hidden_neuron_num_list,
                        hidden_distance=hidden_neuron_distance_list,
                        hidden_bound=hidden_bound_list,
                        hidden_layer_distance=hidden_layer_distance_list,
                        output_layer_distance=donn["output_layer_distance"],
                        output_dim=output_dim)

            y = net(waves)
            for i in range(len(hidden_layers)):
                tape.watch(net.dls[i + 1].x0)
        
        for i in range(len(hidden_layers)):
            dy_dx_ag = tape.gradient(y, net.dls[i + 1].x0)
            print("%-30s" % "dy_dx by tf:", dy_dx_ag[0:5].numpy())


def test_layer():
    n = 28
    with tf.GradientTape(persistent=True) as tape:
        waves = tf.ones([10, n], dtype=tf.complex64)
        tape.watch(waves)
        first_layer = DiffractiveLayer(neuron_number=n, y=0.0, y_next=10.0, mode="phase")
        second_layer = DiffractiveLayer(neuron_number=n, y=10.0, y_next=20.0, mode="phase")
        tape.watch(first_layer.phase)
        tape.watch(first_layer.x0)

        y = first_layer(second_layer.x0, waves)
        y = tf.abs(y)
    dy_dx1_ag = tape.gradient(y, first_layer.phase)
    print("%-30s" % "dy_dx1 by tf:", dy_dx1_ag.numpy())

def test_gradients():
    
    with tf.GradientTape(persistent=True) as tape:
        
        waves = tf.constant([[2.0, 1.0, 0.5], [1.5, 3.0, 0.0]], dtype=tf.complex64)
        print("waves shape: {}".format(waves.shape))
        h_neuron = tf.constant(2 * Const.Lambda0, dtype=tf.float32)
        x1 = tf.constant([1.5, 2.5, 3.5])
        x2 = tf.constant([2.0, 3.0])
        z = tf.constant([2.5])

        tape.watch(x1)
        tape.watch(x2)
        
        xg1 = tf.expand_dims(x1, axis=-1)
        dx = x2 - xg1
        ddx_dx1 = -1
        
        r = tf.sqrt((dx) ** 2 + z ** 2)
        dr_dx1 = -1 * dx / r
        
        w0 = tf.constant(Const.w0, dtype=tf.float32)
        pi_sqrt = tf.constant(np.sqrt(np.pi), dtype=tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        
        theta = tf.acos(z / r)
        dtheta_dx1 = -1 / tf.sqrt(1 - (z / r) ** 2) * (-dr_dx1 * z) / r ** 2
        # theta[np.isnan(theta)] = 0

        w = Const.w0 * tf.sqrt(1 + (z * Const.Lambda / pi / Const.w0 ** 2) ** 2)
        
        E_RSM_coeff = tf.sqrt(2 * w0 / r / pi_sqrt)
        dE_RSM_coeff_dx1 = 1 / 2 / E_RSM_coeff * (-dr_dx1 * 2 * w0 / pi_sqrt / r ** 2) * z / r

        E_RSM = Const.k_RSM * E_RSM_coeff * z / r
        dE_RSM_dx1 = -1.5 * Const.k_RSM * E_RSM_coeff * z / r ** 2.5 * dr_dx1

        E_GBM_coeff = tf.exp(-dx ** 2 / w ** 2)
        dE_GBM_coeff_dx1 = E_GBM_coeff * (-2) * dx / w ** 2 * ddx_dx1

        E_GBM = Const.k_GBM * tf.sqrt(w0 / w) * E_GBM_coeff
        dE_GBM_dx1 = Const.k_GBM * tf.sqrt(w0 / w) * dE_GBM_coeff_dx1
        
        F_coupling = tf.constant(Const.F_coupling_coeff, dtype=tf.complex64)

        phase = tf.constant(Const.TM02_beta, dtype=tf.float32) * h_neuron / 2
        P_propagation = tf.complex(tf.cos(-phase), tf.sin(-phase))

        TM02_phi_decoupling = tf.constant(Const.TM02_phi_decoupling, dtype=tf.float32)
        TM02_eta_decoupling = tf.constant(Const.TM02_eta_decoupling, dtype=tf.complex64)
        k_sub = tf.constant(Const.k_sub, dtype=tf.float32)
        G_decouping = TM02_eta_decoupling * tf.complex(tf.cos(TM02_phi_decoupling), tf.sin(TM02_phi_decoupling)) \
                      * (tf.cast(E_RSM, dtype=tf.complex64) + tf.cast(E_GBM, dtype=tf.complex64)) * tf.complex(tf.cos(-k_sub * r), tf.sin(-k_sub * r))

        dE_dx0 = dE_RSM_dx1 + dE_GBM_dx1
        dG_decouping_dx1 = TM02_eta_decoupling * tf.complex(tf.cos(TM02_phi_decoupling), tf.sin(TM02_phi_decoupling))  \
                            * (tf.cast(dE_dx0, dtype=tf.complex64) * tf.complex(tf.cos(-k_sub * r), tf.sin(-k_sub * r)) \
                            + (tf.cast(E_RSM, dtype=tf.complex64) + tf.cast(E_GBM, dtype=tf.complex64)) * tf.complex(tf.cos(-k_sub * r), tf.sin(-k_sub * r)) \
                            * tf.constant(-1j, dtype=tf.complex64) * tf.cast(k_sub, dtype=tf.complex64) * tf.cast(dr_dx1, dtype=tf.complex64))

        Ex = F_coupling * P_propagation * G_decouping
        dEx_dx1 = F_coupling * P_propagation * dG_decouping_dx1

        y = tf.matmul(waves, Ex)
        print("y shape: {}".format(y.shape))
        dout = tf.ones(y.shape, dtype=tf.complex64)
        dy_dx1 = tf.matmul(tf.transpose(waves), dout) * dEx_dx1
        print("%-30s" % "y by tf:", y.numpy())


        print("Ex shape: {}".format(Ex.shape))
    
    print("%-30s" % "dr_dx1 by numpy", tf.reduce_sum(dr_dx1, axis=-1).numpy())
    dr_dx1_ag = tape.gradient(r, x1)
    print("%-30s" % "dr_dx1 by tf:", dr_dx1_ag.numpy())

    print("%-30s" % "dE_RSM_coeff_dx1 by numpy:", tf.reduce_sum(dE_RSM_coeff_dx1, axis=-1).numpy())
    dE_RSM_coeff_dx1_ag = tape.gradient(E_RSM_coeff, x1)
    print("%-30s" % "dE_RSM_coeff_dx1 by tf:", dE_RSM_coeff_dx1_ag.numpy())

    print("%-30s" % "dE_RSM_dx1 by numpy:", tf.reduce_sum(dE_RSM_dx1, axis=-1).numpy())
    dE_RSM_dx1_ag = tape.gradient(E_RSM_coeff, x1)
    print("%-30s" % "dE_RSM_dx1 by tf:", dE_RSM_dx1_ag.numpy())

    print("%-30s" % "dE_GBM_coeff_dx1 by numpy:", tf.reduce_sum(dE_GBM_coeff_dx1, axis=-1).numpy())
    dE_GBM_coeff_dx1_ag = tape.gradient(E_GBM_coeff, x1)
    print("%-30s" % "dE_GBM_coeff_dx1 by tf:", dE_GBM_coeff_dx1_ag.numpy())

    print("%-30s" % "dE_GBM_dx1 by numpy:", tf.reduce_sum(dE_GBM_dx1, axis=-1).numpy())
    dE_GBM_dx1_ag = tape.gradient(E_GBM, x1)
    print("%-30s" % "dE_GBM_dx1 by tf:", dE_GBM_dx1_ag.numpy())

    print("%-30s" % "dG_decouping_dx1 by numpy:", tf.reduce_sum(tf.math.real(dG_decouping_dx1), axis=-1).numpy())
    dG_decouping_dx1_ag = tape.gradient(G_decouping, x1)
    print("%-30s" % "dG_decouping_dx1 by tf:", dG_decouping_dx1_ag.numpy())

    print("%-30s" % "dEx_dx1 by numpy:", tf.reduce_sum(tf.math.real(dEx_dx1), axis=-1).numpy())
    dEx_dx1_ag = tape.gradient(Ex, x1)
    print("%-30s" % "dEx_dx1 by tf:", dEx_dx1_ag.numpy())

    print("%-30s" % "dy_dx1 by numpy:", tf.reduce_sum(tf.math.real(dy_dx1), axis=-1).numpy())
    dy_dx1_ag = tape.gradient(y, x1)
    print("%-30s" % "dy_dx1 by tf:", dy_dx1_ag.numpy())

    # print(dx2)

if __name__ == '__main__':
    test_DONN()

