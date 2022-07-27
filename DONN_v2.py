"""
Take Json File as DONN Definition
"""

import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
from models.flexible_donn import Flexible_DONN
from models.dummy_donn import Dummy_DONN
import encoding.utils
from utils.solver import Solver
import argparse
import json
import utils.helpers

def main(args):
    if args.mode == "phi":
        phi_init = "default"
    elif args.mode == "x0":
        phi_init = "default"
    else:
        raise ValueError("Invalid mode: %s" % args.mode)
    
    if args.encoding  == "amplitude":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size)
    elif args.encoding == "phase":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size)
        (train_X, train_y), (test_X, test_y) = encoding.utils.phase_encoding(train_X, train_y, test_X, test_y)
    elif args.encoding == "fft":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size, fft=True)
    else:
        raise ValueError("Invalid encoding: %s" % args.encoding)

    input_dim = args.size ** 2
    output_dim = 10

    # read json
    with open(args.json_file, 'r') as f:
        DONNs = json.load(f)["ONN"]
    # parse json


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

        if DONN["dummy"] == True:
            donn = Dummy_DONN(input_neuron_num=input_dim,
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
                            phi_init=phi_init,
                            nonlinear=False
            )
            donn.initial_dummy_cells()
            donn.remove_dummy_cells()
        else:
            donn = Flexible_DONN(input_neuron_num=input_dim,
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
                                phi_init=phi_init,
                                nonlinear=False
            )
        if args.not_plot == False:
            donn.plot_structure(args.checkpoint_name)

        data = {}
        data["X_train"] = train_X
        data["y_train"] = train_y
        data["X_val"] = test_X
        data["y_val"] = test_y

        solver = Solver(donn, data,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    mode=args.mode,
                    verbose=args.verbose,
                    constrained=args.constrained,
                    lr_decay=args.lr_decay,
                    checkpoint_name=args.checkpoint_name,
                    dummy=DONN["dummy"],
                    )
        if args.assessment == True:
            solver.train_assessment(args.num_assess)
        else:
            solver.train()

if __name__ == '__main__':
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
    args = parser.parse_args()

    main(args)
