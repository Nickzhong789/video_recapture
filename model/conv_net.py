import cv2
import numpy as np
import matplotlib.pyplot as pt
    
import operator
import os
import sys
import glob

from random import shuffle

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.utils.data as data_utils


from multiprocessing import Process, freeze_support


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import copy

#Network_implementation

#Kaiming_He weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)




class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96, momentum=0.9),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96, momentum=0.9),
            nn.LeakyReLU(inplace = True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 96, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 2),
            #nn.LeakyReLU(inplace=True), 
            #nn.Linear(1024, 2),
        ) 
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        out = self.features(x)
        self.features.apply(init_weights)
        out = out.reshape(out.size(0), -1)            #flattens image
        out = self.classifier(out)
        out = self.log_softmax(out)
        return out