import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
from utils import constants as Const
import utils.helpers
import models.onn_layer
import models.layers as Layers
import models.flexible_donn as flexible_donn

def generate_donn(input_neuron_num,
                input_neuron_distance,
                hidden_layer_num,
                hidden_neuron_nums,
                hidden_layer_distances,
                hidden_neuron_distances,
                output_neuron_num,
                output_neuron_distance,
                output_layer_distance,
                phi_init,
                nonlinear):

    max_width = 0
    for i in range(hidden_layer_num):
        width = (hidden_neuron_nums[i] + 10) * hidden_neuron_distances[i]
        if width > max_width:
            max_width = width
    

    total_width = max((input_neuron_num + 10) * input_neuron_distance, 
                        max_width,
                        (output_neuron_num + 10) * output_neuron_distance) 
    hidden_bounds = []

    input_bound = utils.helpers.get_bound(input_neuron_num, input_neuron_distance, total_width)
    output_bound = utils.helpers.get_bound(output_neuron_num, output_neuron_distance, total_width)

    for i in range(hidden_layer_num):
        hidden_bounds.append(utils.helpers.get_bound(hidden_neuron_nums[i], hidden_neuron_distances[i], total_width))

    #print(input_Ex)

    test_donn = flexible_donn.Flexible_DONN(input_neuron_num=input_neuron_num,
                                            input_distance=input_neuron_distance,
                                            input_bound=input_bound,
                                            output_neuron_num=output_neuron_num,
                                            output_distance=output_neuron_distance,
                                            output_bound=output_bound,
                                            hidden_layer_num=hidden_layer_num,
                                            hidden_neuron_num=hidden_neuron_nums,
                                            hidden_distance=hidden_neuron_distances,
                                            hidden_bound=hidden_bounds,
                                            hidden_layer_distance=hidden_layer_distances,
                                            output_layer_distance=output_layer_distance,
                                            phi_init=phi_init,
                                            nonlinear=nonlinear
                                            )

    return test_donn

