import models
import os
import random
import time
import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, TensorDataset
from torch.optim import Adam, SGD
import torch.nn as nn
import numpy as np


class MyRB_enc(nn.Module):
    def __init__(self, model_enc, batchNorm=False, f=1):
        super(MyRB_enc, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 32)
        # self.fc = nn.Linear(32, 2)
        self.fc = nn.Linear(32, 1)
        
        
    def forward(self, x, out_conv6):
        out = out_conv6.view(out_conv6.size(0), -1)
        out_fc1 = nn.functional.relu(self.fc1(out))
        out_fc2 = self.drop(nn.functional.relu(self.fc2(out_fc1)))
        out_fc3 = self.drop(nn.functional.relu(self.fc3(out_fc2)))
        out_fc4 = nn.functional.relu(self.fc4(out_fc3))
        out_fc5 = self.fc(out_fc4)
        return nn.functional.sigmoid(out_fc5).squeeze(1)
    
    
class TN_classifier(nn.Module):
    def __init__(self, batchNorm=False):
        super(TN_classifier, self).__init__()
        f = 2
        self.batchNorm = batchNorm
        self.enc = models.TN_enc(batchNorm=self.batchNorm, f=f)
        self.dec = models.TN_dec(self.enc, batchNorm=self.batchNorm, f=f)
        self.classifier = MyRB_enc(self.enc, batchNorm=self.batchNorm, f=f)

    def forward(self, x):
        out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = self.enc.forward(x)
        out0, out1, out2, out3, out4, out5, out6 = self.dec.forward(x,
                                (out_conv1, out_conv2, out_conv3,
                                 out_conv4, out_conv5, out_conv6))
        rb = self.classifier.forward(x, out_conv6)
        return rb, out0, out1, out2, out3, out4, out5, out6