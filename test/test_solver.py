import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import models.donn as donn
import encoding.fft as fft
import encoding.utils
from utils import constants as Const
from utils.solver import Solver

def test_solver(mode):

    # reshaped image is 7 * 7
    new_size = 14
    input_dim = new_size ** 2
    (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(new_size)
    print("train dataset size: ", train_X.shape[0])
    print("test dataset size: ", test_X.shape[0])

    if mode == "phi":
        phi_init = "random"
    elif mode == "x0":
        phi_init = "default"
    data = {}

    t = 1
    hidden_neuron_num = 400 * t
    
    DONN_model = donn.get_donn_example(input_neuron_num=input_dim, 
                                        hidden_layer_num=4,
                                        phi_init=phi_init,
                                        nonlinear=False,
                                        hidden_neuron_num=hidden_neuron_num,
                                        output_neuron_num=10,
                                        input_distance=4 * t,
                                        hidden_distance=3e-6 / Const.Lambda0,
                                        output_distance=80 * t,)
    DONN_model.plot_structure()

    data["X_train"] = train_X
    data["y_train"] = train_y
    data["X_val"] = test_X
    data["y_val"] = test_y

    solver = Solver(DONN_model, data,
                    learning_rate=0.5e-11,
                    num_epochs=20,
                    batch_size=50,
                    mode="x0",
                    verbose=False,
                    constrained=False,
                    lr_decay=0.95,
                    checkpoint_name="x0_checkpoint_4_layer",
                    )
    
    solver.train()

def main():
    test_solver("x0")

if __name__ == '__main__':
    main()
    pass
