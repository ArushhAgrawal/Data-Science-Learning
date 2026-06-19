import torch
from torch import nn
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

num_class=4
num_features=2


#create multiclass data
x_blob,  y_blob = make_blobs(n_samples=1000,
                             n_features=num_features,
                             centers= num_class,
                             cluster_std=1.5, #std is standerd devistion gives the data a little shakeup a mix
                             random_state=42)

#turn into tensors
x_blob, y_blob= torch.Tensor(x_blob).to(torch.float32), torch.from_numpy(y_blob).to(torch.int64)

#spliting data in tran and test
x_train, x_test, y_train, y_test = train_test_split(x_blob,
                                                    y_blob,
                                                    test_size=0.8,
                                                    random_state=32)

#device agnostic code
device="mps" if torch.mps.is_available() else "cpu"
print(device)
#visualisation
plt.figure(figsize=(10,7))
plt.scatter(x_blob[:,0], x_blob[:,1], c=y_blob)
#plt.show()

#making model
model_0= nn.Sequential(nn.Linear(in_features=2, out_features=16),
                       nn.ReLU(),
                       nn.Linear(in_features=16, out_features=8),
                       nn.ReLU(),
                       nn.Linear(in_features=8, out_features=8),
                       nn.ReLU(),
                       nn.Linear(in_features=8, out_features=4)).to(device)
#setting up loss fn and learning rate
loss_fn= torch.nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(params= model_0.parameters(), lr=0.1)

#acc function
def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred.squeeze(), y_true)
    acc=  (correct.sum()/len(y_true) )*100
    return acc

#training and testing loop
epochs=100
torch.manual_seed=32
torch.mps.manual_seed=32
for epoch in range(epochs):
    model_0.train()
    y_logits= model_0(x_train.to(device))
    y_pred= torch.argmax(torch.softmax(y_logits, dim=1),dim=1)
    loss=loss_fn(y_logits, y_train.to(device))
    train_acc= accuracy(y_pred=y_pred.squeeze(), y_true=y_train.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        y_test_logits= model_0(x_test.to(device))
        y_test_pred= torch.argmax(torch.softmax(y_test_logits,dim=1),dim=1)
        test_acc= accuracy(y_pred=y_test_pred.squeeze(), y_true=y_test.to(device))
        loss_test= loss_fn(y_test_logits, y_test.to(device))
print(f"training loss: {loss}, test loss: {loss_test}")
print(f"training acc: {train_acc}, test acc: {test_acc}")
from helper import plot_decision_boundary
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)#row , column , index
plt.title("train")
plot_decision_boundary(model_0, x_train, y_train)
plt.subplot(1,2,2)
plt.title("test")
plot_decision_boundary(model_0, x_test,y_test)
plt.show()

