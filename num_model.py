import torch
import torchvision
from torch import nn
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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
                           batch_size=64,
                           shuffle=False)
img,lab=next(iter(train_dataloader))
print(len(img))