import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import pandas as pd

import time

import models.donn as donn
import encoding.fft as fft
import encoding.utils
from utils import constants as Const




def test_mnist():
    example_num = 100
    (train_X, train_y), (test_X, test_y) = encoding.utils.import_MNIST()
    random_indices = np.random.choice(train_X.shape[0], example_num, replace=False, )
    example_X = train_X[random_indices]
    example_y = train_y[random_indices]
    input_Ex = np.reshape(example_X, (example_X.shape[0], -1))
    print(input_Ex.shape)

def test_fft():
    example_num = 1000
    (train_X, train_y), (test_X, test_y) = encoding.utils.import_MNIST()
    random_indices = np.random.choice(train_X.shape[0], example_num, replace=False, )
    example_X = train_X[random_indices]
    example_y = train_y[random_indices]
    output_dim = 25
    input_Ex = fft.fft_2D(example_X, output_dim ** 0.5)

    print(input_Ex)    


def test_multibatch_train(mode):
    """
        Test training with multiple batches
    """
    batch_size = 20
    new_size = 7
    iter_num = 100

    input_dim = new_size ** 2

    iter_history = []
    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    t = 1
    hidden_neuron_num = 400 * t

    if mode == "phi":
        phi_init = "random"
    elif mode == "x0":
        phi_init = "default"
    
    DONN_model = donn.get_donn_example(input_neuron_num=input_dim, 
                                        hidden_layer_num=2,
                                        phi_init=phi_init,
                                        nonlinear=False,
                                        hidden_neuron_num=hidden_neuron_num,
                                        output_neuron_num=10,
                                        input_distance=15 * t,
                                        hidden_distance=3e-6 / Const.Lambda0,
                                        output_distance=80 * t,)

    # DONN_model.plot_structure()
    for iter in range(iter_num):
        
        (train_X, train_y), (test_X, test_y) = encoding.utils.import_MNIST()
        random_indices = np.random.choice(train_X.shape[0], batch_size, replace=False, )
        example_X = train_X[random_indices]
        example_y = train_y[random_indices]
    
        (compressed_Ex, compressed_y) = encoding.utils.get_training_example(batch_size, new_size)


        preprocess = True
        if preprocess == True:
            input_dim = 49
            input_Ex = fft.fft_2D(example_X, input_dim ** 0.5) / np.max(np.abs(example_X))
            target = example_y
        else:
            input_dim = new_size ** 2
            input_Ex = compressed_Ex
            target = compressed_y
        hidden_neuron_num = 200


        dout = np.ones((batch_size, 10))        
        if mode == "phi":
            output_Ex, loss, dx_list = DONN_model.loss_v3(input_Ex, y=target)
        elif mode == "x0":
            output_Ex, loss, dx_list = DONN_model.loss_v4(input_Ex, y=target)
        # y_inf = np.argmax(np.abs(output_Ex), axis=1)
        # accuracy = np.mean(y_inf == target)

        if mode == "phi":
            lr = 1
        elif mode == "x0":
            lr = 1e-11

        for i in range(DONN_model.layer_num):
            dx = lr * dx_list[i]
            if mode == "phi":
                if iter_num <= 20:
                    print("dphi", i, ":", dx)
                DONN_model.layers[i].phi -= dx
                DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
                DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi
            elif mode == "x0":
                if iter_num <= 20:
                    print("dx0", i, ":", dx / Const.Lambda0)
                DONN_model.layers[i].x0 -= dx
                #x0_outbound_l = DONN_model.layers[i].x0 < DONN_model.layers[i].x0_left_limit
                #x0_outbound_r = DONN_model.layers[i].x0 > DONN_model.layers[i].x0_right_limit
                #DONN_model.layers[i].x0[x0_outbound_l] = DONN_model.layers[i].x0_left_limit[x0_outbound_l]
                #DONN_model.layers[i].x0[x0_outbound_r] = DONN_model.layers[i].x0_right_limit[x0_outbound_r]            #DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
            #DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi

            #print("out_r: ", np.abs(out[0, 0]), np.abs(out[0, 4]), np.abs(out[0, 9]))
        # print("out_r: ", np.abs(out[0, target_index]))
        DONN_model.update_dests()


        # Test 
        if iter_num <= 20:
            print("Ex:", output_Ex[0])
            print("abs:", np.abs(output_Ex[0]))
            print("loss:", loss)


        elif iter % 10 == 0:
            # train accuracy, loss
            output_Ex = DONN_model.forward(input_Ex)
            y_inf_train = np.argmax(np.abs(output_Ex), axis=1)
            train_accuracy = np.mean(y_inf_train == target)

            # test accuracy
            batch_size = 40
            random_indices = np.random.choice(train_X.shape[0], batch_size, replace=False, )
            example_X = train_X[random_indices]
            example_y = train_y[random_indices]
        
            (test_Ex, test_y) = encoding.utils.get_training_example(batch_size, new_size)

            if preprocess == True:
                input_dim = 49
                input_Ex = fft.fft_2D(example_X, input_dim ** 0.5) / np.max(np.abs(example_X))
                target_test = example_y
            else:
                input_dim = new_size ** 2
                input_Ex = test_Ex
                target_test = test_y
            hidden_neuron_num = 200
            
            output_Ex = DONN_model.forward(input_Ex)
            y_inf_test = np.argmax(np.abs(output_Ex), axis=1)
            test_accuracy = np.mean(y_inf_test == target_test)
            loss = np.mean(loss)
            # print("iter %d Ex norm:" % k, np.abs(output_Ex))
            print("iter %d loss:" % iter, loss)
            print("iter %d train accuracy:" % iter, train_accuracy)
            print("target:", target)
            print("predict:", y_inf_train)  
            print("iter %d test accuracy:" % iter, test_accuracy)
            print("target:", target_test)
            print("predict:", y_inf_test)

            iter_history.append(iter)
            loss_history.append(loss.item())
            train_accuracy_history.append(train_accuracy.item())
            test_accuracy_history.append(test_accuracy.item())
            # print("Ex:", output_Ex[0])
            # print("Ex abs:", np.abs(output_Ex[0]))

    if mode == "phi":
        print(DONN_model.layers[0].phi - DONN_model.layers[0].original_phi)
    elif mode == "x0":
        print((DONN_model.layers[0].x0 - DONN_model.layers[0].original_x0) / Const.Lambda0)

    dict = {'iter': iter_history, 'loss': loss_history, 'train_accuracy': train_accuracy_history, 'test_accuracy': test_accuracy_history}
    df = pd.DataFrame(dict) 
    df.to_csv('./data/output/test_train.csv') 

def test_train(mode):
    """
        Test training with Different Modes Support:
            1. phi
            2. x0
    """

    example_num = 10
    new_size = 7
    (train_X, train_y), (test_X, test_y) = encoding.utils.import_MNIST()
    random_indices = np.random.choice(train_X.shape[0], example_num, replace=False, )
    example_X = train_X[random_indices]
    example_y = train_y[random_indices]
    
    (compressed_Ex, compressed_y) = encoding.utils.get_training_example(example_num, new_size)
    if mode == "phi":
        phi_init = "random"
    elif mode == "x0":
        phi_init = "default"

    preprocess = False
    if preprocess == True:
        input_dim = 49
        input_Ex = fft.fft_2D(example_X, input_dim ** 0.5) / np.max(np.abs(example_X))
    else:
        input_dim = new_size ** 2
        input_Ex = compressed_Ex

    hidden_neuron_num = 400

    DONN_model = donn.get_donn_example(input_neuron_num=input_dim, 
                                        hidden_layer_num=2,
                                        phi_init=phi_init,
                                        nonlinear=False,
                                        hidden_neuron_num=hidden_neuron_num,
                                        output_neuron_num=10,
                                        input_distance=15,
                                        hidden_distance=3e-6 / Const.Lambda0,
                                        output_distance=80,)

    DONN_model.plot_structure()
    iter = 1000

    dout = np.ones((example_num, 10))
    target = example_y
    print(target)
    print(input_Ex)
    for k in range(iter):
        
        DONN_model.update_dests()
        if mode == "phi":
            output_Ex, loss, dx_list = DONN_model.loss_v3(input_Ex, y=target)
        elif mode == "x0":
            output_Ex, loss, dx_list = DONN_model.loss_v4(input_Ex, y=target)

        y_inf = np.argmax(np.abs(output_Ex), axis=1)
        accuracy = np.mean(y_inf == target)
        if iter <= 20:
            print("Ex:", output_Ex[0])
            print("abs:", np.abs(output_Ex[0]))
            print("loss:", loss)
            print("accuracy:", accuracy)
        elif k % 20 == 0:
            # print("iter %d Ex norm:" % k, np.abs(output_Ex))
            print("iter %d loss:" % k, np.mean(loss))
            print("iter %d accuracy:" % k, accuracy)
        
        if mode == "phi":
            lr = 1e-1
        elif mode == "x0":
            lr = 1e-11

        for i in range(DONN_model.layer_num):
            dx = lr * dx_list[i]

            if mode == "phi":
                if iter <= 20:
                    print("dphi", i, ":", dx)
                DONN_model.layers[i].phi -= dx
                DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
                DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi
            elif mode == "x0":
                if iter <= 20:
                    print("dx0", i, ":", dx / Const.Lambda0)
                DONN_model.layers[i].x0 -= dx
                #x0_outbound_l = DONN_model.layers[i].x0 < DONN_model.layers[i].x0_left_limit
                #x0_outbound_r = DONN_model.layers[i].x0 > DONN_model.layers[i].x0_right_limit
                #DONN_model.layers[i].x0[x0_outbound_l] = DONN_model.layers[i].x0_left_limit[x0_outbound_l]
                # DONN_model.layers[i].x0[x0_outbound_r] = DONN_model.layers[i].x0_right_limit[x0_outbound_r]            #DONN_model.layers[i].phi[DONN_model.layers[i].phi < 0] = 0 
            #DONN_model.layers[i].phi[DONN_model.layers[i].phi >= 2 * np.pi] = 2 * np.pi
        
        #print("out_r: ", np.abs(out[0, 0]), np.abs(out[0, 4]), np.abs(out[0, 9]))
    # print("out_r: ", np.abs(out[0, target_index]))
    output_Ex = DONN_model.forward(input_Ex)
    y_inf = np.argmax(np.abs(output_Ex), axis=1)
    print("target:", target)
    print("predict:", y_inf)
    print("Ex:", output_Ex[0])
    print("Ex abs:", np.abs(output_Ex[0]))
    if mode == "phi":
        print(DONN_model.layers[0].phi - DONN_model.layers[0].original_phi)
    elif mode == "x0":
        print((DONN_model.layers[0].x0 - DONN_model.layers[0].original_x0) / Const.Lambda0)

def main():
    starttime = time.time()
    test_multibatch_train("x0")
    endtime = time.time()
    print("time:", endtime - starttime)
if __name__ == "__main__":
    main()