from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))


from matplotlib import pyplot as plt
import numpy as np



from models.onn_layer import ONN_Layer
from utils import constants as Const
import encoding.utils
import utils.helpers
from models.donn import DONN

def plot_layer_pattern(layer: ONN_Layer,
                       input_Ex: np.ndarray,
                       figname=None,
                       height=10,
                       nx=1000, 
                       ny=100,
                       mode="norm"):
    """
    Plot Contour Map of One Layer
    Input:
        input_Ex: shape of (n, )
        height: int
            the relative height of plot region
        nx, ny: int
            resolution of the map
        mode: str
            norm, angle, real or imag
        figname: str, optional
            figure save location        
    """

    assert input_Ex.shape[0] == 1, "input_Ex must be contain only one dimension"

    length = layer.length
    height = -height * Const.Lambda0
    y_start = layer.y0
    dests = np.zeros((ny, nx, 2), dtype=np.float64)
    x = np.linspace(0, length, nx)
    y = np.linspace(y_start, y_start + height, ny)
    dests[:, :, 0] = x
    dests[:, :, 1] = y.reshape(ny, 1)
    print(dests.shape)
    dests = dests.reshape(nx * ny, 2)
    Ex = layer.calculate_Ex(input_Ex, dests)
    Ex = Ex.reshape(ny, nx)

    if mode == "norm":
        Ex = np.abs(Ex)
    elif mode == "angle":
        Ex = np.angle(Ex)
    elif mode == "real":
        Ex = np.real(Ex)
    elif mode == "imag":
        Ex = np.imag(Ex)
    else:
        raise ValueError('Wrong mode')

    X, Y = np.meshgrid(x, y)
    cset=plt.contourf(X, Y, Ex, 60, cmap='jet')
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")
    plt.title("Electric field Ex")
    plt.colorbar(cset).set_label('Ex' + mode)
    plt.axis('equal')
    plt.show()

    if figname:
        plt.savefig(figname, format='pdf', bbox_inches='tight')

def plot_layer_cross_section(layer: ONN_Layer,
                             figname=None,
                             y_location=10,
                             nx=1000, 
                             mode="norm"):
    """
    Plot Contour Map of One Layer
    Input:
        y_location: int
            the relative height of plot region
        nx: int
            resolution of the map
        mode: str
            norm, angle, real or imag
        figname: str, optional
            figure save location        
    """    
    dests = np.zeros((nx, 2), dtype=np.float64)
    length = layer.length
    x = np.linspace(0, length, nx)
    dests[:, 0] = np.linspace(0, length, nx)
    dests[:, 1] = -y_location * Const.Lambda0

    Ex, cache = layer.forward_propagation(dests)

    if mode == "norm":
        Ex = np.abs(Ex)
    elif mode == "angle":
        Ex = np.angle(Ex)
    elif mode == "real":
        Ex = np.real(Ex)
    elif mode == "imag":
        Ex = np.imag(Ex)
    else:
        raise ValueError('Wrong mode')
    
    plt.plot(x, Ex)
    plt.xlabel("X (um)")
    plt.title("Electric field Ex")

    if figname:
        plt.savefig(figname, format='pdf', bbox_inches='tight')

    plt.show()


def plot_whole_field(donn: DONN,
                     figname=None,
                     n=10,
                     mode="norm"):
    
    length = donn.input_layer.length
    nx = int(length / Const.Lambda0 * n)
    x = np.linspace(0, length, nx)

    # calculate input layer inside space
    ny = int(donn.input_layer.h_neuron / 2 / Const.Lambda0 * n)
    y = np.linspace(0, donn.input_layer.h_neuron / 2, ny)
    dests = np.zeros((ny, nx, 2), dtype=np.float64)

    dests[:, :, 0] = x
    dests[:, :, 1] = y.reshape(ny, 1)
    Ex = donn.input_layer.calculate_inside_space(dests)
    
    previous_layer = donn.input_layer
    for i in range(donn.hidden_layer_num):
        curr_layer = donn.hidden_layer[i]
        ny = int((donn.layer_distance - previous_layer.h_neuron / 2 - curr_layer.h_neuron / 2) / Const.Lambda0 * n)
        y = np.linspace(previous_layer.y0_coord - previous_layer.h_neuron / 2,
                        -curr_layer.y0_coord + curr_layer.h_neuron / 2,
                        ny)
        dests = np.zeros((ny, nx, 2), dtype=np.float64)
        dests[:, :, 0] = x
        dests[:, :, 1] = y.reshape(ny, 1)
        # calculate previous_layer output field
        Ex, cache = donn.previous_layer.forward_propagation(dests)
        # calculate current_layer inside space

    for hidden_layer_index, hidden_layer in enumerate(donn.hidden_layers):
        ny = int((donn.layer_distance - previous_layer.h_neuron / 2 - hidden_layer.h_neuron / 2) / Const.Lambda0 * n)
        y = np.linspace(previous_layer.y0_coord - previous_layer.h_neuron / 2,
                        -hidden_layer.y0_coord + hidden_layer.h_neuron / 2,
                        ny)



    if mode == "norm":
        Ex = np.abs(Ex)
    elif mode == "angle":
        Ex = np.angle(Ex)
    elif mode == "real":
        Ex = np.real(Ex)
    elif mode == "imag":
        Ex = np.imag(Ex)
    else:
        raise ValueError('Wrong mode')

    X, Y = np.meshgrid(x, y)
    cset=plt.contourf(X, Y, Ex, 60)
    plt.xlabel("X (um)")
    plt.ylabel("Y (um)")
    plt.title("Electric field Ex")
    plt.colorbar(cset).set_label('Ex' + mode)
    plt.show()

def plot_distribution(model, X):
    y = model.forward(X)

def main():
    new_size = 5
    neuron_number = new_size ** 2
    distance = 6
    total_width = (neuron_number + 10) * distance
    input_Ex = encoding.utils.get_input_example(example_num=1, new_size=new_size)

    bound = utils.helpers.get_bound(neuron_number, distance, total_width)
    test_layer = ONN_Layer(neuron_number=new_size ** 2, distance=10, bound=bound)
    plot_layer_pattern(test_layer, input_Ex, height=50, figname="test.pdf")
    #plot_layer_cross_section(test_layer, y_location=15, mode="norm")
    #plot_layer_cross_section(test_layer, y_location=15, mode="real")
    #plot_layer_cross_section(test_layer, y_location=15, mode="angle")

    #donn = models.donn.get_donn_example(example_num=1)
    #plot_whole_field(donn)

if __name__ == '__main__':
    main()