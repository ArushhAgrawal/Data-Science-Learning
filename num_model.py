import torch
import torchvision
from torch import nn
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

#loading data
train_data=datasets.MNIST(root= "number_data",
                          train=True,
                          transform=transforms.ToTensor(),
                          target_transform=None,
                          download=False)
test_data= datasets.MNIST(root= "number_data",
                          train= False,
                          transform=ToTensor(),
                          target_transform=None,
                          download=False)
train_dataloader=DataLoader(dataset=train_data,
                            batch_size=96,
                            shuffle=True)
test_dataloader=DataLoader(dataset=test_data,
                           batch_size=32,
                           shuffle=False)

#device agnostic code
device="mps" if torch.mps.is_available() else "cpu"

#making model
class NumberModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1=nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2)).to(device)
        self.conv_block2= nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2)).to(device)
        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=16, out_features=10)).to(device)
    def forward(self,x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))

model=NumberModel()
