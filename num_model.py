import torch
import torchvision
from torch import nn
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time

#loading data
train_data=datasets.MNIST(root= "number_data",
                          train=True,
                          transform=transforms.ToTensor(),
                          target_transform=None,
                          download=False)
test_data= datasets.MNIST(root= "number_data",
                          train= False,
                          transform=ToTensor(),
                          target_transform=None,
                          download=False)
train_dataloader=DataLoader(dataset=train_data,
                            batch_size=96,
                            shuffle=True)
test_dataloader=DataLoader(dataset=test_data,
                           batch_size=32,
                           shuffle=False)

#device agnostic code
device="mps" if torch.mps.is_available() else "cpu"

#making model
class NumberModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1=nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2)).to(device)
        self.conv_block2= nn.Sequential(nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2)).to(device)
        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.ReLU(),
                                      nn.Linear(in_features=16*7*7, out_features=10)).to(device)
    def forward(self,x):
        return self.classifier(self.conv_block2(self.conv_block1(x)))

model=NumberModel()

#defining loss and optimizer functions
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

#dummy
# torch.manual_seed(32)
# with torch.inference_mode():
#     for batch,(x,y) in enumerate(test_dataloader):
#         y1=model(x.to(device))
#         loss=loss_fn(y1,y.to(device))
#print(loss)

#accuracy function
def accuracy(y_pred,y_true):
    correct= torch.eq(y_pred,y_true.to(device)).sum()
    acc=correct/len(y_true.to(device))*100
    return acc

#train/test loop
torch.manual_seed(32)
torch.mps.manual_seed(32)
epochs=3
start=time.time()
for epoch in range(epochs):
    model.train()
    for batch, (x,y) in enumerate(train_dataloader):
        y_train_logits= model(x.to(device))
        y_pred_train=torch.argmax(y_train_logits, dim=1)
        loss_train=loss_fn(y_train_logits,y.to(device))
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        for batch, (x,y) in enumerate(test_dataloader):
            y_logits_test= model(x.to(device))
            y_pred_test= torch.argmax(y_logits_test, dim=1)
            loss_test= loss_fn(y_logits_test, y.to(device)) 
end=time.time() 
print("train loss:",loss_train,"\ntest loss",loss_test)      
print("time take:", end-start)

