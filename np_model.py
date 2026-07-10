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
x_test= test_data.data.numpy()
x_test=x_test[:10000]
#for labels
y_labels_train= train_data.targets.numpy()
y_labels_train= y_labels_train[:10000]
y_labels_test= test_data.targets.numpy()
y_labels_test= y_labels_test[:10000]
print(y_labels_test.shape)

# def loss(y_pred, y_true):
#     i=1
#     for i in range(10):
#         y_pred[i-1:i]#finding the predicited output
        

