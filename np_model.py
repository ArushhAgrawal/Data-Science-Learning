import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np
from torchvision.transforms import ToTensor
#loadining data
train_data=datasets.MNIST(root= "number_data",
                          train=True,
                          transform=ToTensor,
                          target_transform=None,
                          download=False)
test_data= datasets.MNIST(root= "number_data",
                          train= False,
                          transform=transforms.ToTensor,
                          target_transform=None,
                          download=False)

#conversion to np
x_train= train_data.data.numpy()/255.0
x_train=x_train[:10000]
x_train= x_train.reshape(10000,784)
x_test= test_data.data.numpy()/255.0 
x_test=x_test[:1000]
x_test= x_test.reshape(1000,784)
#for labels
y_labels_train= train_data.targets.numpy()
y_labels_train= y_labels_train[:10000]
y_labels_test= test_data.targets.numpy()
y_labels_test= y_labels_test[:1000]
# print(y_labels_test.shape)

def params(out_features):
    weight= np.random.random((784,out_features))*0.01
    bias= np.random.random((out_features))
    return weight, bias

def passes(x_data , weight, bias):
    z= np.matmul(x_data,weight)+bias
    a=np.maximum(0,z)
    return a,z

def loss(y_pred, y_true):
    m=y_pred.shape[0]
    target= np.zeros_like(y_pred)
    target[np.arange(m),y_true]=1#its going to convert say there is a zero array it will find the location from m and y_true say 3 so at index 3 it will become 1
    cost= np.mean((y_pred- target)**2)
    return cost

def backward(x_data, weight, bias, a, z, y_true ):#a is relu z and z is x@weight+bias
    m = x_data.shape[0]
    target= np.zeros_like(a)
    target[np.arange(m),y_true]=1
    dL_da = 2 * (a - target) / 10       
    relu_grad = (z > 0).astype(float)#avoid boolean   
    dL_dz = dL_da * relu_grad                  
    dW = np.matmul(x_data.T,  dL_dz) / m                  
    db = np.sum(dL_dz, axis=0) / m             
    return dW, db

def update(weight, bias, dW,db, lr):
    weight-= lr*dW
    bias-= lr*db
    return weight,  bias

#train/test loop
weight, bias= params(10)
lr=0.01
epochs=100
for epoch in range(epochs):
    a,z = passes(x_train, weight, bias)
    loss_train= loss(a,  y_labels_train)
    dw, db= backward(x_train, weight, bias, a,z,y_labels_train)
    weight, bias= update(weight, bias, dw, db, lr)
    if epoch % 10 ==0:
        print("train" , loss_train)
epoch=0 
for epoch in range(10):
    a,z = passes(x_test, weight, bias)
    loss_test = loss(a, y_labels_test)
print(f"test, {loss_test}")