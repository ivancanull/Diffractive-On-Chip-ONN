# Diffractive-On-Chip-ONN

## TensorFlow version
Networks defined in ``./tf/onn.py``.

For regular network, run ``python ./tf/onn_solver.py ``. For dummy network, run ``python ./tf/onn_solver_custom_training.py ``. The input arguments are the same.

Network structure defined in ``./json`` JSON files

For example: ``python ./tf/onn_solver_custom_training.py --json_file ./json/example.json --learning_rate 1e-8 --dataset MNIST ``

Arguments:

- **size**: Dimension of input images.
- **learning_rate**: Recommended learning rate: 1e-2 for phase mode, 1e-8 for x0 mode.
- **mode**: Modulation mode. "x0" for changing location of neurons, "phase" for changing phase after each layer.
- **encoding**: Encoding method: phase, amplitude, or fft. Currently, amplitude encoding is used.
- **json_file**: Network sturcture definition. 
**Note**: In json file, the length, width are times of lambda0, for example, ``"neuron_distance": 12,`` means the neuron distance is 12 * lambda0.

