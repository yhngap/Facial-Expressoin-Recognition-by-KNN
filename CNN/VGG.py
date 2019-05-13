import torch as torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,num_classes=7):
        super(VGG16,self).__init__()
        self.features = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(16,16,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(16,32,kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),

             nn.Conv2d(32,32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, kernel_size=3, padding=1),
             nn.ReLU(inplace=True)
         )

        self.classifier = nn.Sequential(
            nn.Linear(56448,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048,num_classes))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

def test_vgg():
    net = VGG16()
    y = net(torch.rand(32, 1, 42, 42))
    print(y.size())

