from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os
import cv2
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

import encoding.utils


def fft_2D(imgs: np.ndarray, 
           output_dim: int) -> np.ndarray:
    """
    Convert Images to Ex input. Please pay attention that output_dim = 2 * k + 1
    
    Args:
        imgs: m images of dimension (n * n), total shape is (m, n, n) or (n, n, n)
        output_dim: squared root of the output fft dimension, also the input neuron 
                    numbers of the DONN

    Returns: 
        input_Ex: encoded input Ex of (m, k) or (k, )
    """
    assert (output_dim - 1) % 2 == 0

    fft2_result = np.fft.fft2(imgs)
    shifted_fft = np.fft.fftshift(fft2_result,  axes=(1, 2, ))

    rows = imgs.shape[-2] // 2
    cols = imgs.shape[-1] // 2
    new_size = int((output_dim - 1) // 2)
    masked_shifted_fft = shifted_fft[:, rows - new_size : rows + new_size + 1, cols - new_size : cols + new_size + 1]
    input_Ex = masked_shifted_fft.reshape(masked_shifted_fft.shape[:-2] + (-1, ))
    return input_Ex

def main():
    
    # get original images
    example_num = 4
    example_input = encoding.utils.get_MNIST_example(example_num)
    new_size = 10
    compressed_input = np.zeros((example_input.shape[0], new_size, new_size), dtype=np.float64)
    for i in range(example_num):
        compressed_input[i] = cv2.resize(example_input[i], (new_size, new_size))
    print(compressed_input.shape)
    
    for i in range(example_num):
        plt.subplot(221 + i)
        plt.imshow(compressed_input[i], cmap=plt.get_cmap('gray'))
    
    # 
    plt.savefig("example.png")
    plt.show()
    fft2_result = np.fft.fft2(example_input)
    new_size = 5
    resized_shifted_fft = np.fft.fftshift(fft2_result[:, 0:new_size, 0:new_size], axes=(1, 2, ))
    print(resized_shifted_fft.shape)
    
    rows = example_input.shape[-2] // 2
    cols = example_input.shape[-1] // 2
    
    new_size = 2

    shifted_fft = np.fft.fftshift(fft2_result,  axes=(1, 2, ))
    
    # the center is at rows / 2, therefore the range is  rows / 2 - k to rows / 2 + k + 1
    masked_shifted_fft = shifted_fft[:, rows - new_size : rows + new_size + 1, cols - new_size : cols + new_size + 1]
    print(masked_shifted_fft)
    magnitude_spectrum = 20 * np.log(np.abs(masked_shifted_fft))
    for i in range(example_num):
        plt.subplot(221 + i)
        plt.imshow(magnitude_spectrum[i], cmap=plt.get_cmap('gray'))
    # plt.show()
    plt.savefig("fft.png")
    plt.show()

if __name__ == '__main__':
    main()