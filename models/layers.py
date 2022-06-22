from ctypes.wintypes import tagRECT
from tkinter import Y
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import encoding.utils
import models.donn as donn


def norm_layer_forward(x):
    y = np.abs(x)
    x_real = np.real(x)
    x_imag = np.imag(x)

    cache = {}
    cache['real'] = x_real
    cache['imag'] = x_imag
    cache['norm'] = y

    return y, cache

def norm_layer_backward(dout, cache):

    
    x_real = cache['real']
    x_imag = cache['imag']
    y = cache['norm']
    # print ("x_real / y shape: ", (x_real / y).shape)
    dout_r = dout * x_real / y
    dout_i = dout * x_imag / y

    return dout_r, dout_i

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
        x: Input data, of shape (N, C) where x[i, j] is the score for the jth
           class for the ith input.
        y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
           0 <= y[i] < C

    Returns a tuple of:
        loss: Scalar giving the loss
        dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def mean_square_error(m, g, g_true=1000, g_false=50):
    """
    Computes the mean square error between output plane intensity m and target g

    Inputs:
        m: Output plane intensity of shape (N, C)
        g: Vector of labels, of shape (N,)
    
    Returns a tuple of:
        loss: Scalar giving the loss
        dx: Gradient of the loss with respect to x
    """
    N = m.shape[0]
    target = np.ones(m.shape) * g_false ** 2
    target[np.arange(N), g] = g_true ** 2
    s = np.abs(m) ** 2
    loss = np.mean((s - target) ** 2, axis=0)
    s_minus_g = s - target
    return loss, s_minus_g

def normalized_mean_square_error(m, g, g_true=1, g_false=0, non_linear=False, non_linear_threshold=0.1):
    """
    Computes the mean square error between output plane intensity m and target g

    Inputs:
        m: Output plane intensity of shape (N, C)
        g: Vector of labels, of shape (N,)
        g_true: The target value for true g
        g_false: The target value for false g
    
    Returns a tuple of:
        loss: Scalar giving the loss
        S: sum of s
        s_minus_g: s / S - g
        coeff: (s_k / S - g_k) * (S - s_k) / (S ^ 2)
    """
    N = m.shape[0]
    # suppress wrong numbers
    target = np.ones(m.shape) * g_false
    # do not supress wrong numbers
    # target = m
    # target[np.arange(N), g] = g_true
    s = np.abs(m) ** 2
    S = np.sum(s, axis=1, keepdims=True)
    s_norm = s / S
    # target = np.copy(s_norm)

    # if use non linear function, we set those backward derivatives with value lower than threshold to be 0
    if non_linear == True:
        target = np.copy(s_norm)
        target[target > non_linear_threshold] = non_linear_threshold

    target[np.arange(N), g] = g_true
    # print(s_norm)
    # print(target)
    coeff = (s_norm - target) * (S - s) / S ** 2
    # print(coeff)
    # s / sum of s - g
    loss = np.mean((s_norm - target) ** 2, axis=1)
    return loss, coeff



def test_target():
    batch_size = 100
    new_size = 5
    (X, y) = encoding.utils.get_training_example(batch_size=batch_size, new_size=new_size)
    s = np.zeros((X.shape[0], 10))
    mean_square_error(s, y)

def main():
    batch_size = 100
    new_size = 5
    (X, y) = encoding.utils.get_training_example(batch_size=batch_size, new_size=new_size)
    X = X / 1e10
    test_donn = donn.get_donn_example(new_size=new_size)
    y_pred = test_donn.forward(X)
    y_norm = np.abs(y_pred) 
    print (y_norm[0])
    print("y_pred shape: ", y_norm.shape)
    print("y shape: ", y.shape)
    loss, dx = softmax_loss(y_norm, y)
    print("loss: ", loss)
    print("dx shape: ", dx.shape)
    print(dx[0])

    np.savetxt("./temp/dx.csv", dx)

if __name__ == '__main__':
    test_target()