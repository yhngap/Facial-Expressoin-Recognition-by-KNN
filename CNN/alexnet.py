import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=7):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(64, 192, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def test_alexnet():
    net = AlexNet()
    y = net(torch.rand(32, 1, 42,42))
    print(y.size())