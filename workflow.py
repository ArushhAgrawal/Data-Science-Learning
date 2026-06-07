#pytorch workflow
#what work flow has
#1 preparing of data
#2 build model
#3 fitting the model to data or training
#4 making prediction and making changes in weights
#5 saving and repeating
import torch
from torch import nn #nn contains all the pytorch building block from nural network
import matplotlib.pyplot as plt

#data preparing and loading
#data can be a lot of things like- Excel sheet, images, videos, audio...etc

#in machine learning - u take data convert to number(vectors) then make computer learn patterns in it.
#to showcase this we create ome leaner regression formula (y=mx+c) to draw a straight line from the known parameters

#create known parameters
weight= 0.7
bias= 0.3

#create data
start=0
end=1
step=0.02
x= torch.arange(start,end,step).unsqueeze(1)#it adds extra dimesnions will be usefull later 
y= weight * x+bias

#print(x[:10])
#print(y[:10])
#print(len(x))
#print(len(y))

#splitting data in training and testing data
#create train,test split
train_split = int(0.8*len(x))
x_train, y_train= x[:train_split], y[:train_split]
x_test, y_test= x[train_split:], y[train_split:]
print(f"traing split:  {train_split}")
print(f"traing x:  {len(x_train)}")
print(f"traing y:  {len(y_train)}")
print(f"testing x:  {len(y_test)}")
print(f"testing y:  {len(y_test)}")