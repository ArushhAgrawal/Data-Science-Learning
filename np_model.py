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
x_train= train_data.data.numpy()
x_train=x_train[:10000]
x_train= x_train.reshape(10000,784)
x_test= test_data.data.numpy()
x_test=x_test[:1000]
x_test= x_test.reshape(1000,784)
#for labels
y_labels_train= train_data.targets.numpy()
y_labels_train= y_labels_train[:10000]
y_labels_test= test_data.targets.numpy()
y_labels_test= y_labels_test[:1000]
# print(y_labels_test.shape)

def loss(y_pred, y_true):
    i=0
    cost= np.array([])
    for i in range(10):
        if i== y_true:
            cost= np.append(cost, (y_pred[y_true]- 1)**2)
        else:
            cost= np.append(cost, (y_pred[i]-0)**2)
    return cost.mean()

weight= np.random.random((784,10))
print(weight.shape)

bias= np.random.random((10))
print(bias)
