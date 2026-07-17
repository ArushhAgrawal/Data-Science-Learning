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
    def __init__(self, in_channels,  out_channels, features=[64,128,256,512]):
        super().__init__()
        self.up= nn.ModuleList()
        self.down= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        #down part
        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels=feature
        
        #up part
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2))#this will make the model move up
            self.up.append(DoubleConv(feature*2, feature))#this will do the conv part up 2 conv like this combined
        
        #bottom part
        self.bottom= DoubleConv(in_features=features[-1], out_features=features[-1]*2)#in feature then out feature 512*2
        self.final_conv= nn.Conv2d(features[0], out_channels,  kernel_size=1)
    def forward(self, x):
        skip=[]
        #downward pass
        for index in self.down:
            x=index(x)
            skip.append(x)#this is the part where it will skip the connection from the heightest to the lowest connection
            x=self.pool(x)#160x160-> 80x80
        x=self.bottom(x)
        skip= skip[::-1]#reversing the list 
        for index in range (0,len(self.up), 2):
            x=self.up[index](x)#up sampling
            skip_tensor= skip[index//2]#divides and then rounds to nearest whole number
            if skip_tensor.shape == x.shape:
                concat_skip= torch.cat((skip_tensor, x),dim=1)    
            else:
                l= x.shape[2]
                w=x.shape[3]
                concat_skip= torch.cat((skip_tensor[:,:, :l, :w], x),dim=1)
            x= self.up[index+1](concat_skip)
        return self.final_conv(x)

#test
x=torch.randn((3,1,180,180))
model= UNET(in_channels=1, out_channels=1)
pred= model(x)
print(pred.shape)
print(x.shape)

