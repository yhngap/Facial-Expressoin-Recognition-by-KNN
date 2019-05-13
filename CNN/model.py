from __future__ import print_function, division
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(1),

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256,momentum=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048,7))


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_of_features(x))
        x = self.classifier(x)
        return x

    def num_of_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def test_mymodel():
    net=Model()
    outputs = net(torch.rand(32, 1, 42, 42))
    print(outputs.size())