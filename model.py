from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

from IPython import embed

class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):

        super(Net, self).__init__()
        self.resnet = models.wide_resnet50_2(pretrained = True)
        # self.resnet = models.resnet50(pretrained = True)
        numFeatures = self.resnet.fc.out_features

        self.layer2 = nn.Linear(numFeatures, 512)
        self.layer3 = nn.Linear(512, num_classes)



    def forward(self, x):
        # Resnet
        x = self.resnet(x)

        # Linear Relu Linear Relu Linear Softmax
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        # x = F.softmax(x)

        return x
