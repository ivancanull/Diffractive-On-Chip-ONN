"""
Take Json File as DONN Definition
Built with PyTorch
"""

import sys, os

from utils import dataset
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))


import numpy as np
import torch
import argparse
import json
import csv

import encoding.utils
import utils.helpers

import torch.nn.functional as F

# from models.flexible_donn import Flexible_DONN
# from models.dummy_donn import Dummy_DONN
from models.onn_torch import DiffractiveNetwork
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from torchvision import transforms
from tqdm import tqdm

def train(model, data, args):

    transform = transforms.ToTensor()
    target_transform = transforms.ToTensor()

    train_dataset = CustomDataset(data["X_train"], data["y_train"])
    val_dataset = CustomDataset(data["X_val"], data["y_val"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,)

    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
        print('Model: "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        else:
            with open(args.result_record_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # model.cuda()

    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):
        log = [epoch]

        model.train()

        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0
        tk0 = tqdm(train_dataloader, ncols=100, total=int(len(train_dataloader)))

        for train_iter, train_data_batch in enumerate(tk0):
            
            X, y = train_data_batch
            labels = F.one_hot(y, num_classes=10).float()
            outputs = model.forward(X)

            # Print model's state_dict
            print("Model's state_dict:")
            for i in range(model.layer_num):
                print("layer %d" % i, model.layers[i].x0[0:10].tolist())

            train_loss_ = criterion(outputs, labels)
            train_counter_ = torch.eq(torch.argmax(labels, dim=1), torch.argmax(outputs, dim=1)).float().sum()

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

            train_len += len(labels)
            train_running_loss += train_loss_.item()
            train_running_counter += train_counter_.item()

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

            tk0.set_description_str('Epoch {}/{} : Training'.format(epoch, args.start_epoch + 1 + args.num_epochs - 1))
            tk0.set_postfix({'Train_Loss': '{:.5f}'.format(train_loss), 'Train_Accuracy': '{:.5f}'.format(train_accuracy)})

        log.append(train_loss)
        log.append(train_accuracy)

        with torch.no_grad():

            model.eval()

            val_len = 0.0
            val_running_counter = 0.0
            val_running_loss = 0.0

            tk1 = tqdm(val_dataloader, ncols=100, total=int(len(val_dataloader)))
            for val_iter, val_data_batch in enumerate(tk1):

                X, y = val_data_batch
                labels = F.one_hot(y, num_classes=10).float()
                outputs = model.forward(X)

                val_loss_ = criterion(outputs, labels)
                val_counter_ = torch.eq(torch.argmax(labels, dim=1), torch.argmax(outputs, dim=1)).float().sum()

                val_len += len(labels)
                val_running_loss += val_loss_.item()
                val_running_counter += val_counter_.item()

                val_loss = val_running_loss / val_len
                val_accuracy = val_running_counter / val_len

                tk1.set_description_str('Epoch {}/{} : Validating'.format(epoch, args.start_epoch + 1 + args.num_epochs - 1))
                tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})

            log.append(val_loss)
            log.append(val_accuracy)

        torch.save(model.state_dict(), (args.model_save_path + str(epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(epoch) + args.model_name + '" saved.')

        with open(args.result_record_path, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)


def main(args):
    if args.mode == "phi":
        phi_init = "default"
    elif args.mode == "x0":
        phi_init = "default"
    else:
        raise ValueError("Invalid mode: %s" % args.mode)
    
    if args.encoding  == "amplitude":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size, compact=args.compact_decoding)
    elif args.encoding == "phase":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size, compact=args.compact_decoding)
        (train_X, train_y), (test_X, test_y) = encoding.utils.phase_encoding(train_X, train_y, test_X, test_y)
    elif args.encoding == "fft":
        (train_X, train_y), (test_X, test_y) = encoding.utils.load_data(args.size, fft=True, compact=args.compact_decoding)
    else:
        raise ValueError("Invalid encoding: %s" % args.encoding)

    

    input_dim = args.size ** 2
    output_dim = args.output_dim

    # read json
    with open(args.json_file, 'r') as f:
        DONNs = json.load(f)["ONN"]
    # parse json


    for DONN_index, DONN in enumerate(DONNs):
        hidden_neuron_num_list = []
        hidden_neuron_distance_list = []
        hidden_bound_list = []
        hidden_layer_distance_list = []
    
        print("---------DONN No. %d---------" % DONN_index)
        input_neuron_distance = DONN["input_neuron_distance"]
        
        max_width = (input_dim + 10) * input_neuron_distance
        hidden_layers = DONN["hidden_layers"]

        for layer_index, layer in enumerate(hidden_layers):
            hidden_neuron_num_list.append(layer["neuron_number"])
            hidden_neuron_distance_list.append(layer["neuron_distance"])
            hidden_layer_distance_list.append(layer["layer_distance"])
            max_width = max((layer["neuron_number"] + 10) * layer["neuron_distance"], max_width)
        
        output_neuron_distance = DONN["output_neuron_distance"]
        max_width = max((output_dim + 10) * output_neuron_distance, max_width)

        input_bound = utils.helpers.get_bound(input_dim, input_neuron_distance, max_width)
        for layer_index, layer in enumerate(hidden_layers):
            hidden_bound = utils.helpers.get_bound(layer["neuron_number"], layer["neuron_distance"], max_width)
            hidden_bound_list.append(hidden_bound)
        output_bound = utils.helpers.get_bound(output_dim, output_neuron_distance, max_width)

        donn_model = DiffractiveNetwork(input_neuron_num=input_dim,
                            input_distance=input_neuron_distance,
                            input_bound=input_bound,
                            output_neuron_num=output_dim,
                            output_distance=output_neuron_distance,
                            output_bound=output_bound,
                            hidden_layer_num=len(hidden_layers),
                            hidden_neuron_num=hidden_neuron_num_list,
                            hidden_distance=hidden_neuron_distance_list,
                            hidden_bound=hidden_bound_list,
                            hidden_layer_distance=hidden_layer_distance_list,
                            output_layer_distance=DONN["output_layer_distance"],
                            phi_init=phi_init,
                            nonlinear=False,
                            compact_decoding=args.compact_decoding,
        )

        if args.not_plot == False:
            # donn_model.plot_structure(args.checkpoint_name)
            pass
        
        data = {}
        print(train_X[0])
        data["X_train"] = torch.Tensor(train_X)
        data["y_train"] = torch.LongTensor(train_y)
        data["X_val"] = torch.Tensor(test_X)
        data["y_val"] = torch.LongTensor(test_y)

        if data["X_train"].dtype == torch.float:
            data["X_train"] = torch.complex(data["X_train"], torch.zeros_like(data["X_train"]))
        if data["X_val"].dtype == torch.float:
            data["X_val"] = torch.complex(data["X_val"], torch.zeros_like(data["X_val"]))
        
        train(donn_model, data, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', dest='size', default=14, type=int, help='dimension of the input image')
    parser.add_argument('--mode', dest='mode', default='x0', type=str, help='mode of the configuration')
    parser.add_argument('--encoding', dest='encoding', default='amplitude', type=str, help='encoding of the input signal')

    parser.add_argument('--json_file', dest='json_file', default='./json/example.json', type=str, help='structure definition json file')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-11, type=float, help='learning rate of the DONN')
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int, help='the number of epochs to train the model')
    parser.add_argument('--start_epoch', dest='start_epoch', default=0, type=int, help='start from which epoch')
    parser.add_argument('--batch_size', dest='batch_size', default=50, type=int, help='batch size of the model')
    parser.add_argument('--verbose', dest='verbose', default=False, type=bool, help='print=')
    parser.add_argument('--constrained', dest='constrained', default=False, type=bool, help='with constrained neurons location')
    parser.add_argument('--lr_decay', dest='lr_decay', default=0.95, type=float, help='learning rate decay')
    parser.add_argument('--checkpoint_name', dest='checkpoint_name', default='temp', type=str, help='checkpoint_name')

    parser.add_argument('--not_plot', action='store_true', dest='not_plot', help='do not plot the structure of DONN')
    parser.add_argument('--assessment', action='store_true', dest='assessment', help='assess the DONN structure')
    parser.add_argument('--num_assess', dest='num_assess', default=100, type=int, help='number of iterations during struture assessment')

    parser.add_argument('--compact_decoding', action='store_true', dest='compact_decoding', help='decoding in a compact way')
    parser.add_argument('--output_dim', dest='output_dim', default=10, type=int, help='dimension of the output decoder')

    parser.add_argument('--model_name', dest='model_name', type=str, default='_model.pth')
    parser.add_argument('--model_save_path', dest='model_save_path', type=str, default="./saved_model/")
    parser.add_argument('--result_record_path', dest='result_record_path', type=str, default="./results/result.csv", help="directory of result records")
    parser.add_argument('--whether_load_model', dest='whether_load_model', default=False, type=bool, help='whether load model to continue training')
    args = parser.parse_args()

    main(args)
