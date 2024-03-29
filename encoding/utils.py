import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
import cv2
import os.path
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import encoding.fft

mapping = {0: [0, 1],
           1: [0, 2],
           2: [0, 3],
           3: [1, 2],
           4: [0, 4],
           5: [1, 3],
           6: [2, 3],
           7: [1, 4],
           8: [2, 4],
           9: [3, 4]}

def convert_y_to_compact(y):
    dim = y.shape[0]
    new_y = np.zeros((dim, 5), dtype=int)
    for i in range(dim):
        new_y[i][mapping[y[i]]] = 1
    return new_y
        
def convert_compact_to_y(y_c):
    dim = y_c.shape[0]
    y = np.zeros((dim, 1), dtype=int)

def load_data(new_size, fft=False, compact=False, dataset="MNIST"):

    if dataset == "MNIST":
        (train_X, train_y), (test_X, test_y) = import_MNIST()
    elif dataset == "Fashion_MNIST":
        (train_X, train_y), (test_X, test_y) = import_Fashion_MNIST()
    else:
        raise ValueError("Dataset not supported")

    #example_input = get_MNIST_example(batch_size)
    if fft == True:
        out_file = "./data/%s/%s_fft_" % (dataset, dataset) + str(new_size) + "x" + str(new_size) + ".npz"
    else:
        out_file = "./data/%s/%s_" % (dataset, dataset) + str(new_size) + "x" + str(new_size) + ".npz"

    if os.path.isfile(out_file):
        npz_file = np.load(out_file)

        # compact decoding
        if compact == True:
            return (npz_file['train_X'], convert_y_to_compact(npz_file['train_y'])), (npz_file['test_X'], convert_y_to_compact(npz_file['test_y']))
        else:
            return (npz_file['train_X'], npz_file['train_y']), (npz_file['test_X'], npz_file['test_y'])
    else:
        dim = train_X.shape[0]
        compressed_train_X = np.zeros((dim, new_size, new_size))
        if fft == True:
            compressed_train_X = encoding.fft.fft_2D(train_X, new_size)
        else:
            for i in range(dim):
                compressed_train_X[i] = cv2.resize(train_X[i], (new_size, new_size))
        compressed_train_X = np.reshape(compressed_train_X, (dim, new_size ** 2))
        
        dim = test_X.shape[0]
        compressed_test_X = np.zeros((dim, new_size, new_size))
        if fft == True:
            compressed_test_X = encoding.fft.fft_2D(test_X, new_size)
        else:
            for i in range(dim):
                compressed_test_X[i] = cv2.resize(test_X[i], (new_size, new_size))
        compressed_test_X = np.reshape(compressed_test_X, (dim, new_size ** 2))
        
        # compact decoding
        if compact == True:
            train_y = convert_y_to_compact(train_y)
            test_y =  convert_y_to_compact(test_y)
        
        with open(out_file, 'wb') as f:
            np.savez(f, train_X=compressed_train_X, train_y=train_y, test_X=compressed_test_X, test_y=test_y)
        return (compressed_train_X, train_y), (compressed_test_X, test_y)

def import_MNIST():
    """
    Import MNIST from tf.keras
    
    Return: 
        data: (train_x, train_y, test_x, test_y)
    """

    data_file = "./data/MNIST/MNIST.npz"
    if os.path.isfile(data_file):
        npz_file = np.load(data_file)
        return (npz_file['train_X'], npz_file['train_y']), (npz_file['test_X'], npz_file['test_y'])
    else:
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        with open(data_file, 'wb') as f:
            np.savez(f, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
        return (train_X, train_y), (test_X, test_y) 

def import_Fashion_MNIST():
    """
    Import Fashion MNIST from tf.keras

    Return:
        data: (train_x, train_y), (test_x, test_y)
    
    """

    data_file = "./data/Fashion_MNIST/Fashion_MNIST.npz"
    if os.path.isfile(data_file):
        npz_file = np.load(data_file)
        return (npz_file['train_X'], npz_file['train_y']), (npz_file['test_X'], npz_file['test_y'])
    else:
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
        with open(data_file, 'wb') as f:
            np.savez(f, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)
        return (train_X, train_y), (test_X, test_y) 

def get_MNIST_example(example_num):
    (train_X, train_y), (test_X, test_y) = import_MNIST()
    shape = train_X.shape
    random_indices = np.random.choice(shape[0], example_num, replace=False, )
    example_x = train_X[random_indices]
    return example_x

def get_input_example(example_num, new_size):

    example_input = get_MNIST_example(example_num)
    compressed_input = np.zeros((example_input.shape[0], new_size, new_size), dtype=np.float64)
    for i in range(example_num):
        compressed_input[i] = cv2.resize(example_input[i], (new_size, new_size))
    input_Ex = np.reshape(compressed_input, (example_num, new_size ** 2))
    return input_Ex

def get_training_example(batch_size, new_size, random_indices=True):
    """
    Get Training Dataset with Batch Size and Input Neuron Size and

    Input:
        batch_size: the batch size of the training dataset
        new_size: the new size of the input images, the input neuron 
                  number is new_size ** 2

    Return:
        sampled_X of shape (batch_size, new_size ** 2)
        sampled_y of shape (batch_size, )
    """
    (train_X, train_y), (test_X, test_y) = import_MNIST()

    shape = train_X.shape
    if random_indices == True:
        random_indices = np.random.choice(shape[0], batch_size, replace=False, )
        sampled_X = train_X[random_indices]
        sampled_y = train_y[random_indices]
    else:
        indices = np.arange(batch_size)
        sampled_X = train_X[indices]
        sampled_y = train_y[indices]

    #example_input = get_MNIST_example(batch_size)
    
    compressed_input = np.zeros((sampled_X.shape[0], new_size, new_size), dtype=np.float64)
    for i in range(batch_size):
        compressed_input[i] = cv2.resize(sampled_X[i], (new_size, new_size))
    compressed_Ex = np.reshape(compressed_input, (batch_size, new_size ** 2))
    return (compressed_Ex, sampled_y)

def phase_encoding(train_X, train_y, test_X, test_y):
    train_X = np.exp((train_X / 255.0) * 2 * np.pi * 1j)
    test_X = np.exp((test_X / 255.0) * 2 * np.pi * 1j)
    return (train_X, train_y), (test_X, test_y)

def test():
    (train_X, train_y), (test_X, test_y) = import_MNIST()
    shape = train_X.shape
    example_num = 4
    random_indices = np.random.choice(shape[0], example_num, replace=False, )
    example_x = train_X[random_indices]
    #example_y = train_y[random_indices]
    
    for i in range(example_num):
        plt.subplot(221 + i)
        plt.imshow(example_x[i], cmap=plt.get_cmap('gray'))
    plt.show()
    (compressed_Ex, sampled_y) = get_training_example(4, 10)

def test_convert():
    (train_X, train_y), (test_X, test_y) = import_MNIST()
    new_y = convert_y_to_compact(test_y)[0:10]
    print(new_y)
    c_y = np.zeros(new_y.shape, dtype=int)
    indices = np.argpartition(new_y, -2, axis=1)[:, -2:]
    c_y[np.expand_dims(np.arange(10,), axis=1), indices] = 1
    print(indices)
    print(np.sum((c_y == new_y), axis=1) == 5)

def main():
    (train_X, train_y), (test_X, test_y) = import_Fashion_MNIST()

if __name__ == '__main__':
    main()