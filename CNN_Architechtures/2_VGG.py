from statistics import mode
from turtle import forward
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_Net(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super(VGG_Net,self).__init__()
        self.in_channels = in_channels
        self.cov_layers = self.create_conv_layers(VGG_types['VGG16'])
        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = self.cov_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x
    
    def create_conv_layers(self,architechture):
        layers=[]
        in_channels = self.in_channels

        for x in architechture:
            if(type(x))==int:
                out_channels = x
                layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                        nn.BatchNorm2d(x),
                        nn.ReLU()]
                in_channels = x
            elif x=='M':
                layers+= [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        return nn.Sequential(*layers)

def test_VGG_Net():
    model = VGG_Net(in_channels=3,num_classes=1000)
    x = torch.randn(1,3,224,224)
    output = model(x)
    print(output.shape)

if __name__ == "__main__":
    test_VGG_Net()