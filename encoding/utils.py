import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import os.path


def load_data(new_size):
    (train_X, train_y), (test_X, test_y) = import_MNIST()


    #
    #example_input = get_MNIST_example(batch_size)
    
    out_file = "./data/MNIST/MNIST_" + str(new_size) + "x" + str(new_size) + ".npz"


    if os.path.isfile(out_file):
        npz_file = np.load(out_file)
        return (npz_file['train_X'], npz_file['train_y']), (npz_file['test_X'], npz_file['test_y'])
    else:
        dim = train_X.shape[0]
        compressed_train_X = np.zeros((dim, new_size, new_size))
        for i in range(dim):
            compressed_train_X[i] = cv2.resize(train_X[i], (new_size, new_size))
        compressed_train_X = np.reshape(compressed_train_X, (dim, new_size ** 2))
        
        dim = test_X.shape[0]
        compressed_test_X = np.zeros((dim, new_size, new_size))
        for i in range(dim):
            compressed_test_X[i] = cv2.resize(test_X[i], (new_size, new_size))
        compressed_test_X = np.reshape(compressed_test_X, (dim, new_size ** 2))
        
        with open(out_file, 'wb') as f:
            np.savez(f, train_X=compressed_train_X, train_y=train_y, test_X=compressed_test_X, test_y=test_y)
        return (compressed_train_X, train_y), (compressed_test_X, test_y)
        

def import_MNIST():
    """
    Import Keras from tf.keras
    
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
    #
    #example_input = get_MNIST_example(batch_size)
    
    compressed_input = np.zeros((sampled_X.shape[0], new_size, new_size), dtype=np.float64)
    for i in range(batch_size):
        compressed_input[i] = cv2.resize(sampled_X[i], (new_size, new_size))
    compressed_Ex = np.reshape(compressed_input, (batch_size, new_size ** 2))
    return (compressed_Ex, sampled_y)


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
    
def main():
    import_MNIST()

if __name__ == '__main__':
    main()