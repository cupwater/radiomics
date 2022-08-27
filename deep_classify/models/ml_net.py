'''
Author: Peng Bo
Date: 2022-08-11 21:17:49
LastEditTime: 2022-08-12 00:51:17
Description: 

'''
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['MLNet']

class MLNet(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(MLNet, self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
