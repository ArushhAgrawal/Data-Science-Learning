#cnn are known as convolutional nural network
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data= datasets.FashionMNIST(root= "image_data",
                                  train=True,
                                  transform=torch.Tensor(),
                                  target_transform=None)
test_data= datasets.FashionMNIST(root="image_data",
                                 train=False,
                                 transform=torch.Tensor(),
                                 target_transform=None)
                                
train_dataloader= DataLoader(dataset=train_data,
                            batch_size=96,
                            shuffle=True)
test_dataloader= DataLoader(dataset=test_data,
                            batch_size=32,
                            shuffle=True)
#device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"

class FashionModel(nn.Module):
    def __init__ (self):
        super().__init__()
        self.conv_block1= nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,padding=1,stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1,stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2 )#stide is by default set to match the kernel value
                                        )      
        self.conv_block2=nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1,stride=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,padding=1,stride=1),
                                       nn.MaxPool2d(kernel_size=2, stride=2))         
        self.classifier= nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features=16*0, out_features=10))
    def forward(self, x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))
torch.manual_seed(32)
model_2=FashionModel()
print(model_2)
