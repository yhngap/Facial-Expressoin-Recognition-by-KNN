from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(64, momentum=0.5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn_conv4 = nn.BatchNorm2d(128, momentum=0.5)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn_conv5 = nn.BatchNorm2d(256, momentum=0.5)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn_conv6 = nn.BatchNorm2d(512, momentum=0.5)

        self.fc1 = nn.Linear(in_features=512, out_features=4096)
        self.bn_fc1 = nn.BatchNorm1d(4096, momentum=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=2048)
        self.bn_fc2 = nn.BatchNorm1d(2048, momentum=0.5)

        self.fc25 = nn.Linear(in_features=2048, out_features=1024)
        self.bn_fc25 = nn.BatchNorm1d(1024, momentum=0.5)

        self.fc3 = nn.Linear(in_features=1024, out_features=7)

    def forward(self, x):
        x = self.bn_x(x)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv1(self.conv1(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv2(self.conv2(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv3(self.conv3(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv4(self.conv4(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv5(self.conv5(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.leaky_relu(self.bn_conv6(self.conv6(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.2)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = F.leaky_relu(self.bn_fc25(self.fc25(x)))
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def test_mymodel():
    net=Model()
    outputs = net(torch.rand(32, 1, 42, 42))
    print(outputs.size())