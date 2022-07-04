import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import models.donn as donn
import models.flexible_donn as flexible_donn
import encoding.fft as fft
import encoding.utils
from utils import constants as Const
from utils.solver import Solver
import argparse

def main(args):

    if args.mode == "phi":
        phi_init = "random"
        raise ValueError("Don't use mode: %s" % args.mode)
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
    if args.two_stage == True:
        DONN_model = flexible_donn.get_local_global_donn_example(input_neuron_num=input_dim,
                                                                output_neuron_num=args.output_neuron_num,
                                                                local_layer_num=args.local_layer_num,
                                                                local_neuron_num=args.local_neuron_num,
                                                                global_layer_num=args.global_layer_num,
                                                                global_neuron_num=args.global_neuron_num,
                                                                input_distance=4,
                                                                local_neuron_distance=3e-6 / Const.Lambda0,
                                                                global_neuron_distance=3e-6 / Const.Lambda0,
                                                                local_layer_distance=15e-6 / Const.Lambda0,
                                                                global_layer_distance=1500e-6 / Const.Lambda0,
                                                                output_distance=80,
                                                                phi_init=phi_init,
                                                                nonlinear=False,)
    else:
        DONN_model = donn.get_donn_example(input_neuron_num=input_dim, 
                                            hidden_layer_num=args.hidden_layer_num,
                                            phi_init=phi_init,
                                            nonlinear=False,
                                            hidden_neuron_num=args.hidden_neuron_num,
                                            output_neuron_num=args.output_neuron_num,
                                            # currently not support design space exploration yet
                                            input_distance=4,
                                            hidden_distance=3e-6 / Const.Lambda0,
                                            output_distance=80,)

    data = {}
    data["X_train"] = train_X
    data["y_train"] = train_y
    data["X_val"] = test_X
    data["y_val"] = test_y

    solver = Solver(DONN_model, data,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                mode=args.mode,
                verbose=args.verbose,
                constrained=args.constrained,
                lr_decay=args.lr_decay,
                checkpoint_name=args.checkpoint_name,
                )
    
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', dest='size', default=14, type=int, help='dimension of the input image')
    parser.add_argument('--mode', dest='mode', default='x0', type=str, help='mode of the configuration')
    parser.add_argument('--encoding', dest='encoding', default='amplitude', type=str, help='encoding of the input signal')
    parser.add_argument('--two_stage', action='store_true', dest='two_stage', help='enable two-stage DONN')

    parser.add_argument('--hidden_layer_num', dest='hidden_layer_num', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--hidden_neuron_num', dest='hidden_neuron_num', default=400, type=int, help='number of hidden neurons')
    parser.add_argument('--output_neuron_num', dest='output_neuron_num', default=10, type=int, help='number of output neurons')

    parser.add_argument('--local_layer_num', dest='local_layer_num', default=2, type=int, help='number of local layers')
    parser.add_argument('--local_neuron_num', dest='local_neuron_num', default=400, type=int, help='number of local neurons')
    parser.add_argument('--global_layer_num', dest='global_layer_num', default=3, type=int, help='number of global layers')
    parser.add_argument('--global_neuron_num', dest='global_neuron_num', default=400, type=int, help='number of global neurons')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.5e-11, type=float, help='learning rate of the DONN')
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int, help='the number of epochs to train the model')
    parser.add_argument('--batch_size', dest='batch_size', default=50, type=int, help='batch size of the model')
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='print=')
    parser.add_argument('--constrained', dest='constrained', default=False, type=bool, help='with constrained neurons location')
    parser.add_argument('--lr_decay', dest='lr_decay', default=1.0, type=float, help='learning rate decay')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='temp', type=str, help='checkpoint_name')
    args = parser.parse_args()

    main(args)
