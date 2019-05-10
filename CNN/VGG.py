
import torch
import torch.nn as nn
import torch as torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,num_classes=7):
        super(VGG16,self).__init__()
        self.features = nn.Sequential(
             nn.Conv2d(1,32,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32,32,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(32,64,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )

        self.classifier = nn.Sequential(
            nn.Linear(112896,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def test_vgg():
    net = VGG16()
    y = net(torch.rand(32, 1, 42, 42))
    print(y.size())

