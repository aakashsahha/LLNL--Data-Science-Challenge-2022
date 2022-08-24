# importing the libraries
import pandas as pd
import numpy as np
import os

# for reading and displaying images
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, Subset

from img_util import GaussianFilter, Voxelizer3D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py



# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self,use_cuda=True, verbose = 0):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(19, 64, 7, 2, 0)
        self.conv_layer2 = self._conv_layer_set(64, 128, 5, 2, 0)
        self.conv_layer3 = self._conv_layer_set(128, 256, 3, 1, 0)
        self.conv_layer4 = self._conv_layer_set(256, 512, 3, 1, 0)
        
        self.max_pool1 = nn.MaxPool3d(2)
        
        self.max_pool2 = nn.MaxPool3d(2)
        #self.max_pool3 = nn.MaxPool3d(2)
        #self.max_pool4 = nn.MaxPool3d(2)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2048, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def _conv_layer_set(self, in_c, out_c, k_size, stride, padding):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size = k_size, stride= stride, padding=padding, bias=True),
        nn.ReLU(),
        nn.BatchNorm3d(out_c))
        
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.max_pool1(out)
        
        print(out.shape)
        out = self.conv_layer2(out)
        out = self.max_pool2(out)
        out = self.drop1(out)
        
        print(out.shape)
        out = self.conv_layer3(out)
        #out = self.max_pool3(out)


        print(out.shape)
        out = self.conv_layer4(out)
        #out = self.max_pool4(out)

        print(out.shape)
        out = out.view(out.size(0), -1)
        
        print(out.shape)
        out = self.relu1(self.fc1(out))
        out = self.drop2(out)

        out_final = self.sigmoid(self.fc2(out))

        return out_final, out






