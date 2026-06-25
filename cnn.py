#cnn are known as convolutional nural network
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data= datasets.FashionMNIST(root= "image_data",
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  target_transform=None)
test_data= datasets.FashionMNIST(root="image_data",
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 target_transform=None)
                                
train_dataloader= DataLoader(dataset=train_data,
                            batch_size=96,
                            shuffle=True)
test_dataloader= DataLoader(dataset=test_data,
                            batch_size=32,
                            shuffle=False)
images, label= next(iter(train_dataloader))
# print(images.shape)

#device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"

class FashionModel(nn.Module):
    def __init__ (self):
        super().__init__() 
        self.conv_block1= nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3,padding=1,stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3,padding=1,stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2 )#stride is by default set to match the kernel value
                                        ).to(device)    
        self.conv_block2=nn.Sequential(nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3,padding=1,stride=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3,padding=1,stride=1),
                                       nn.MaxPool2d(kernel_size=2, stride=2)).to(device)      
        self.classifier= nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features=16*7*7, out_features=10)
                                       ).to(device)
    def forward(self, x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))

#instantiating model
model= FashionModel()
#defing loss and optimizer functions
loss_fn= nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

#DUMMY MODEL
# torch.manual_seed(32)
# with torch.inference_mode():
#     for batch,(x,y) in enumerate(test_dataloader):
#         y1=model(x)
#         loss=loss_fn(y1,y)
# print(y1.shape)

#defining accuracy
def accuracy(y_pred, y_true):
    correct= torch.eq(y_pred,y_true.to(device)).sum()
    acc= correct/len(y_true.to(device))*100
    return acc
#training loop
torch.manual_seed(32)
torch.mps.manual_seed(32)
epochs=3
for epoch in range(epochs):
    acc_test=0
    acc_train=0
    model.train()
    for batch, (x,y) in enumerate(train_dataloader):
        y_logits_train=model(x.to(device))
        y_pred_train= torch.argmax(y_logits_train, dim=1)
        loss_train= loss_fn(y_logits_train, y.to(device))
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        acc_train+= accuracy(y_pred= y_pred_train, y_true=y)
    avg_acc_train=acc_train/len(train_dataloader)
    model.eval()
    for batch, (x,y) in enumerate(test_dataloader):
        with torch.inference_mode():
            y_logits_test= model(x.to(device))
            y_pred_test= torch.argmax(y_logits_test, dim=1)
            loss_test= loss_fn(y_logits_test, y.to(device))
            acc_test+= accuracy(y_pred= y_pred_test, y_true=y)
    avg_acc_test=acc_test/len(test_dataloader)
print(f"train loss: {loss_test} \nand test  {loss_train}")
print(f"Average accuracy of train: {avg_acc_train} \nand test {avg_acc_test}")