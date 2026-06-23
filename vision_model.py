import torch
from torch import nn
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np
import torchvision 
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

#getting a dataset FashionMnist 
train_data= datasets.FashionMNIST(root= "image_data",
                                  train=True,#we want which type of dataset 
                                  download=True,
                                  transform=transforms.ToTensor(),
                                  target_transform=None)
test_data=datasets.FashionMNIST(root="image_data",
                                train=False,
                                download=True,
                                transform=transforms.ToTensor(),
                                target_transform=None)
header= train_data.classes
header_idx= train_data.class_to_idx
print(f"header index: {header_idx}")
images, labels= train_data[0]
print(f"image shape: {images.shape}")#colour channel, height, width 

#visulisation
# plt.figure(figsize=(10,7))
# plt.imshow(images.squeeze())
# plt.show()

#ploting random image(for exprementation and testing)
torch.manual_seed(32)
fig =  plt.figure(figsize=(9,9))
r,c =4,3
label=[]
for i in range(1,r*c+1):
    random_idx=torch.randint(0, len(train_data), size=[1]).item()#0 is starting range and len(train_data) is the ending,  size converts data into tensor with 1 value and item pops out that number as a regulat int
    images ,label= train_data[random_idx]
    fig.add_subplot(r,c,i)#allows to plot multiple images in a single graph sheet
    plt.imshow(images.squeeze())
#plt.show()

#prepare data loader
#it converts our data from pytorch tensor to python itrable
#we will convert our data to batch size (it will reduce the load on ram)
train_dataloader= DataLoader(dataset=train_data,
                             batch_size=128,
                             shuffle=True)
test_dataloader= DataLoader(dataset=test_data,
                            batch_size=32,
                            shuffle=True)
print(f"loaded data: {train_dataloader}, {test_dataloader}")
train_feature, train_label= next(iter(train_dataloader))
test_feature, test_label= next(iter(test_dataloader))

#device agnostic code
device ="mps" if torch.mps.is_available() else "cpu"

#making model
class FashionModel(nn.Module):
    def __init__ (self):
        super().__init__()
        self.layer_stack= nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=784, out_features= 512),
                                        nn.ReLU(),
                                        nn.Linear(in_features=512, out_features=256),
                                        nn.ReLU(),
                                        nn.Linear(in_features=256, out_features=128),
                                        nn.ReLU(),
                                        nn.Linear(in_features=128, out_features=64),
                                        nn.ReLU(),
                                        nn.Linear(in_features=64, out_features=10)).to(device)
    def forward(self,x):
        return self.layer_stack(x)

model=FashionModel()
#setting up loss function and optimizer
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(params= model.parameters(),
                            lr=0.001)
#defining accuracy function
def accuracy(y_pred, y_true):
    correct= torch.eq(y_pred, y_true.to(device)).sum()
    acc= (correct/len(y_pred))*100
    return acc

#training and testing loop
epochs=3
torch.manual_seed(32)
for epoch in range(epochs):
    model.train()
    for batch, (x,y) in enumerate(train_dataloader):
        y_logits= model(x.to(device))  
        y_train_pred= torch.argmax(y_logits, dim=1)
        loss_train= loss_fn(y_logits, y.to(device))  
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        acc_train= accuracy(y_pred= y_train_pred, y_true= y)
    model.eval()
    with torch.inference_mode():
        for batch, (x,y) in enumerate(test_dataloader):
            y_test_logits=model(x.to(device))
            y_test_pred= torch.argmax(y_test_logits,dim=1)
            loss_test= loss_fn(y_test_logits, y.to(device))
            acc_test= accuracy(y_pred= y_test_pred, y_true= y)
print(f"training loss: {loss_train}, testing loss:{loss_test} ")
print(f"training accuracy: {acc_train}, testing accuracy: {acc_test}")

#make prediction and get model prediction
def eval_model(model: torch.nn.Module,# this shows the type model is torcn.nn.MOdule its not a imp to write
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy):
    loss, acc= 0,0
    model.eval()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            y_logits= model(x.to(device))
            loss= loss_fn(y_logits,y.to(device))
            acc= accuracy(y_true=y,y_pred= torch.argmax(y_logits,dim=1))
            return loss.item(), acc.item()

#testing of eval_model function
model_testing=eval_model(model=model,
                         data_loader=test_dataloader,
                         loss_fn=loss_fn,
                         accuracy=accuracy)    
print(model_testing)    