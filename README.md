# Diffractive-On-Chip-ONN

## TensorFlow version
Networks defined in ``./tf/onn.py``

Run ``python ./tf/onn_solver.py ``

Arguments:

- **size**: Dimension of input images.
- **mode**: Modulation mode. "x0" for changing location of neurons, "phase" for changing phase after each layer.
- **encoding**: Encoding method. Currently, amplitude encoding is used.
- **json_file**: network sturcture definition.
**Note**: In json file, the length, width are times of lambda0, for example, ``"neuron_distance": 12,`` means the neuron distance is 12 * lambda0.
- **others**: learning rate, num_epochs, etc. 

  
