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
    forward_pass= np.maximum(0, np.matmul(x_data,weight)+bias)
    return forward_pass

def loss(y_pred, y_true):
    i=0
    cost= np.array([])
    for i in range(10):
        if i== y_true:
            cost= np.append(cost, (y_pred[y_true]- 1)**2)
        else:
            cost= np.append(cost, (y_pred[i]-0)**2)
    return cost.mean()

def backward(x_data, weight, bias, a, z, target ):
    m = x_data.shape[0]

    dL_da = 2 * (a - target) / 10       
    relu_grad = (z > 0).astype(float)#avoid boolean   
    dL_dz = dL_da * relu_grad                  
    dW = np.matmul(x_data.T,  dL_dz) / m                  
    db = np.sum(dL_dz, axis=0) / m             
    return dW, db

def update(weight, bias, dW,db, lr):
    weight= -lr*dW
    bias= -lr*db
    return weight,  bias


    


