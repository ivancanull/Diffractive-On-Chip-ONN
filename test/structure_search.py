import sys, os
import json

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import models.donn as donn
import models.flexible_donn as flexible_donn
import encoding.fft as fft
import encoding.utils
from utils import constants as Const
from utils.solver import Solver
import argparse
import utils.generator

def test_layer_distance(args):

    (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size)

    input_neuron_num = args.size ** 2
    # layer_distance = [40, 400, 4000]
    
    layer_distance_1 = [4000]
    layer_distance_2 = [80, 100, 130]
    layer_distance_3 = [400, 500, 700]
    hidden_neuron_nums = [420] * 2
    hidden_neuron_distances = [4] * 2

    phi_init="default"
    data = {}
    data["X_train"] = train_X
    data["y_train"] = train_y
    data["X_val"] = test_X
    data["y_val"] = test_y

    best_train_accs = []
    best_val_accs = []
    for L1 in range(len(layer_distance_1)):
        for L2 in range(len(layer_distance_2)):
            for L3 in range(len(layer_distance_3)):
                checkpoint_name = "test_layer_distance" + str(L1) + str(L2) + str(L3)
                hidden_layer_distances = [layer_distance_1[L1], layer_distance_2[L2]]
                donn = utils.generator.generate_donn(input_neuron_num,
                                                    input_neuron_distance=8,
                                                    hidden_layer_num=2,
                                                    hidden_neuron_nums=hidden_neuron_nums,
                                                    hidden_layer_distances=hidden_layer_distances,
                                                    hidden_neuron_distances=hidden_neuron_distances,
                                                    output_neuron_num=10,
                                                    output_neuron_distance=80,
                                                    output_layer_distance=layer_distance_3[L3],
                                                    phi_init=phi_init,
                                                    nonlinear=False)
                
                if args.not_plot == False:
                    donn.plot_structure(checkpoint_name)
                solver = Solver(donn, data,
                            learning_rate=args.learning_rate,
                            num_epochs=args.num_epochs,
                            batch_size=args.batch_size,
                            mode=args.mode,
                            verbose=args.verbose,
                            constrained=args.constrained,
                            lr_decay=args.lr_decay,
                            checkpoint_name=checkpoint_name,
                            )
                
                (best_t, best_train_acc, best_val_acc) = solver.train_assessment(args.num_assess)
                best_train_accs.append(best_train_acc)
                best_val_accs.append(best_val_acc)
    
    print('--------------------------------')
    i = 0
    for L1 in range(3):
        for L2 in range(3):
            for L3 in range(3):
                print('L1: %i L2: %i L3: %i: best_train_acc: %f, best_val_acc: %f' % (layer_distance_1[L1], layer_distance_2[L2], layer_distance_3[L3], best_train_accs[i], best_val_accs[i]))
                i = i + 1
    print('--------------------------------')

def main(args):
    test_layer_distance(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', dest='size', default=14, type=int, help='dimension of the input image')
    parser.add_argument('--mode', dest='mode', default='x0', type=str, help='mode of the configuration')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.5e-11, type=float, help='learning rate of the DONN')
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int, help='the number of epochs to train the model')
    parser.add_argument('--batch_size', dest='batch_size', default=50, type=int, help='batch size of the model')
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='print=')
    parser.add_argument('--constrained', dest='constrained', default=False, type=bool, help='with constrained neurons location')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.95, type=float, help='learning rate decay')

    parser.add_argument('--not_plot', action='store_true', dest='not_plot', help='do not plot the structure of DONN')
    parser.add_argument('--num_assess', dest='num_assess', default=500, type=int, help='number of iterations during struture assessment')

    args = parser.parse_args()
    main(args)
    
