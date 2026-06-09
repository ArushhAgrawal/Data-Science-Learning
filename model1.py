#creating a model that learns the pattern in y=mx+c equation then we give it equation and it gives us output after it learns 
import torch
from torch import nn #nn contains all the pytorch building block from nural network
import matplotlib
matplotlib.use('MACOSX')#this is important if u are on vs code use TkAgg if on windows if on mac use MACOSX it looks good to idk why
import matplotlib.pyplot as plt


#create a linear regression model class
class LinearRegressionModel(nn.Module):#nn.Module- its the building block of pytorch
    def __init__(self):#self is used to store variable inside a class
        super().__init__()
        self.weights= nn.Parameter(torch.rand(1, #we start with random weight and bias at start and then try to adjust it to original weight
                                              requires_grad=True,#it will find the if we increase the w or decrese the weight what change it will cause in the loss(how wrong our answer is)
                                              dtype=torch.float))#nn.parameter - it holds all the weights and bias

        self.bias= nn.Parameter(torch.rand(1,
                                requires_grad=True,
                                dtype=torch.float))
        #forward meathod to define the computation in model(the math your model does to make predicent predictions)
        def forward(self, x: torch.tensor) -> torch.Tensor:#x is the training data
            return self.weights * x + self.bias #this is linear regression formula
        