import torch 
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.convstack= nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU()
        )
    def forward(self,x):
        return self.convstack(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3,  out_features=1, features=[64,128,256,512]):
        self.up= nn.ModuleList()
        self.down= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        #down part
        for feature in feature:
            self.down.append(DoubleConv(in_channels, feature))
        
        #up part
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2))#this will make the model move up
            self.up.append(DoubleConv(feature*2, feature))#this will do the conv part up 2 conv like this combined
                           

