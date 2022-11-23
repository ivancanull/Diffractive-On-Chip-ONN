import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))



import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd

import encoding.utils
import utils.helpers
import utils.plot
from utils import constants as Const
from onn import DONN

tf.autograph.set_verbosity(10)
def main(args):

    # read json
    with open(args.json_file, 'r') as f:
        DONNs = json.load(f)["ONN"]
    # parse json

    for DONN_index, donn in enumerate(DONNs):

        # prepare data for training
        if donn["mode"] == "phase":
            phi_init = "default"
        elif donn["mode"] == "x0":
            phi_init = "default"
        else:
            raise ValueError("Invalid mode: %s" % donn["mode"])
        
        if donn["encoding"]  == "amplitude":
            (x_train, y_train), (x_test, y_test) = encoding.utils.load_data(donn["input_size"], compact=args.compact_decoding, dataset=args.dataset)
        elif donn["encoding"] == "phase":
            (x_train, y_train), (x_test, y_test) = encoding.utils.load_data(donn["input_size"], compact=args.compact_decoding, dataset=args.dataset)
            (x_train, y_train), (x_test, y_test) = encoding.utils.phase_encoding(x_train, y_train, x_test, y_test)
        elif donn["encoding"] == "fft":
            (x_train, y_train), (x_test, y_test) = encoding.utils.load_data(donn["input_size"], fft=True, compact=args.compact_decoding, dataset=args.dataset)
        else:
            raise ValueError("Invalid encoding: %s" % donn["encoding"])

        x_val, y_val = x_train[-10000:], y_train[-10000:]
        x_train, y_train = x_train[:-10000], y_train[:-10000]

        x_train = tf.convert_to_tensor(x_train, dtype=tf.complex64)
        x_val = tf.convert_to_tensor(x_val, dtype=tf.complex64)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.complex64)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(args.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(args.batch_size)
        
        input_dim = donn["input_size"] ** 2
        output_dim = args.output_dim

        hidden_neuron_num_list = []
        hidden_neuron_distance_list = []
        hidden_bound_list = []
        hidden_layer_distance_list = []
    
        print("---------DONN No. %d---------" % DONN_index)
        input_neuron_distance = donn["input_neuron_distance"]
        
        max_width = (input_dim + 10) * input_neuron_distance
        hidden_layers = donn["hidden_layers"]

        for layer_index, layer in enumerate(hidden_layers):
            hidden_neuron_num_list.append(layer["neuron_number"])
            hidden_neuron_distance_list.append(layer["neuron_distance"])
            hidden_layer_distance_list.append(layer["layer_distance"])
            max_width = max((layer["neuron_number"] + 10) * layer["neuron_distance"], max_width)
        
        output_neuron_distance = donn["output_neuron_distance"]
        max_width = max((donn["output_neuron_number"] + 10) * output_neuron_distance, max_width)

        input_bound = utils.helpers.get_bound(input_dim, input_neuron_distance, max_width)
        for layer_index, layer in enumerate(hidden_layers):
            hidden_bound = utils.helpers.get_bound(layer["neuron_number"], layer["neuron_distance"], max_width)
            hidden_bound_list.append(hidden_bound)
        output_bound = utils.helpers.get_bound(donn["output_neuron_number"], output_neuron_distance, max_width)

        if args.compare: 
            MLP_model = tf.keras.Sequential()
            MLP_model.add(tf.keras.Input(shape=(input_dim,)))
            for layer_index, layer in enumerate(hidden_layers):
                MLP_model.add(tf.keras.layers.Dense(layer["neuron_number"], activation="relu"))
            MLP_model.add(tf.keras.layers.Dense(donn["output_neuron_number"]))
            # MLP_model.add(tf.keras.layers.Softmax())
            
            MLP_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, ),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
            
            MLP_model.fit(x=train_dataset,
                    epochs=args.num_epochs,
                    validation_data=val_dataset,)

            MLP_model.evaluate(x=test_dataset)

        
        model = DONN(input_neuron_num=input_dim,
                    input_distance=input_neuron_distance,
                    input_bound=input_bound,
                    output_neuron_num=donn["output_neuron_number"],
                    output_distance=output_neuron_distance,
                    output_bound=output_bound,
                    hidden_layer_num=len(hidden_layers),
                    hidden_neuron_num=hidden_neuron_num_list,
                    hidden_distance=hidden_neuron_distance_list,
                    hidden_bound=hidden_bound_list,
                    hidden_layer_distance=hidden_layer_distance_list,
                    output_layer_distance=donn["output_layer_distance"],
                    output_dim=output_dim,
                    mode=donn["mode"])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, ),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        if not os.path.exists("./temp"):
            os.mkdir("./temp")
        if not os.path.exists("./temp/tfckpt"):
            os.mkdir("./temp/tfckpt")
        if not os.path.exists("./temp/tfcsv"):
            os.mkdir("./temp/tfcsv")
        if not os.path.exists("./temp/weights"):
            os.mkdir("./temp/weights")

        checkpoint_path = "temp/tfckpt/"
        checkpoint_path += donn["name"]

        # create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor="val_accuracy",
                                                         save_weights_only=True,)
        log_file = "./temp/tfcsv/" + donn["name"] + ".csv"
        weight_file = "./temp/weights/" + donn["name"] + ".csv"
        csv_logger = tf.keras.callbacks.CSVLogger(log_file, separator=',', append=True)


        if not os.path.exists(log_file):
            with open(log_file, 'w') as fp:
                pass
        
        if not os.path.exists(weight_file):
            with open(weight_file, 'w') as fp:
                pass


        model.fit(x=train_dataset,
                epochs=args.num_epochs,
                validation_data=val_dataset,
                callbacks=[cp_callback, csv_logger])

        model.evaluate(x=test_dataset)

        model.save_weights('./temp/tfckpt/' + donn["name"])
        
        utils.plot.plot_history(history_file=log_file, filename=donn["name"]+'.pdf')
        df = model.get_weights()
        df.to_csv(weight_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--size', dest='size', default=28, type=int, help='dimension of the input image')
    # parser.add_argument('--mode', dest='mode', default='x0', type=str, help='mode of the configuration')
    # parser.add_argument('--encoding', dest='encoding', default='amplitude', type=str, help='encoding of the input signal')

    parser.add_argument('--json_file', dest='json_file', default='./json/example.json', type=str, help='structure definition json file')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-8, type=float, help='learning rate of the DONN')
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int, help='the number of epochs to train the model')
    parser.add_argument('--batch_size', dest='batch_size', default=50, type=int, help='batch size of the model')

    parser.add_argument('--dataset', dest='dataset', default='MNIST', type=str, help='dataset to train')
    # parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='print=')
    # parser.add_argument('--constrained', dest='constrained', default=False, type=bool, help='with constrained neurons location')
    # parser.add_argument('--lr_decay', dest='lr_decay', default=0.95, type=float, help='learning rate decay')
    # parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='temp', type=str, help='checkpoint_name')

    # parser.add_argument('--not_plot', action='store_true', dest='not_plot', help='do not plot the structure of DONN')
    # parser.add_argument('--assessment', action='store_true', dest='assessment', help='assess the DONN structure')
    # parser.add_argument('--num_assess', dest='num_assess', default=100, type=int, help='number of iterations during struture assessment')
    
    # double decoding
    parser.add_argument('--compact_decoding', action='store_true', dest='compact_decoding', help='decoding in a compact way')
    parser.add_argument('--output_dim', dest='output_dim', default=10, type=int, help='dimension of the output decoder')
    parser.add_argument('--compare', action='store_true', dest='compare', help='compare with MLP models')
    args = parser.parse_args()

    main(args)
