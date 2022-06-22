import sys, os

from pandas import test
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np

from utils import constants as Const
import utils.helpers
import utils.plot as plot
import models.onn_layer
import models.layers as Layers
import models.donn as donn
from models.onn_layer import ONN_Layer
from solver import Solver 

import encoding.utils
import cv2

def main():
    DONN_model = donn.get_donn_example(new_size = 8, hidden_layer_num=1)
    



if __name__ == '__main__':
    main()