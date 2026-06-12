#creating a model that learns the pattern in y=mx+c equation then we give it equation and it gives us output after it learns 
import torch
from torch import nn #nn contains all the pytorch building block from nural network
import matplotlib
matplotlib.use('MacOSX')#this is important if u are on vs code use TkAgg if on windows if on mac use MACOSX it looks good to idk why
import matplotlib.pyplot as plt
from pathlib import Path#required to save model in python

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

#visualisation of real vs prediction
def plot_prediction(train_data= x_train,
                    train_label=y_train,
                    test_labels= y_test,
                    test_data= x_test,
                    prediction=None):
    plt.figure(figsize=(10,7))

    #plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label= "training data")

    #plot testing data in green ur choice
    plt.scatter(test_data, test_labels, c='g', label= "test data")

    #are there prediction
    if prediction is not None:
        plt.scatter(test_data, prediction, c= "red", label= "prediction data")
    
    #show the legends (can put a mirror too)
    plt.legend(prop={"size":14})
    plt.show()#very important to see the graph

#create a linear regression model class
class LinearRegressionModel(nn.Module):#nn.Module- its the building block of pytorch
    def __init__(self):#self is used to store variable inside a class(u can use any allies in place of self)
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
#checking the content of our pytorch model
#create a random seed(we used it so the output we get everytime is fixed not changing on everytime we run)
torch.manual_seed(42)#changing manual seed values affect the predictions too
 
#create subclass
model_0 = LinearRegressionModel()

#check out the parameters
print(list(model_0.parameters()))
#to get a betteroutput in organised way use state_dict()
print(f"organised output {model_0.state_dict()}")

#making prediciton using torch.infrence_mode()(some people still use torch.no_grad but it has les functions than infrece it will be visible later)
with torch.inference_mode():#prediction=infrence not using  will not give any diffrence but later it will be usefull
    y_preds= model_0(x_test)

#print(y_preds)
#plot_prediction(prediction=y_preds)

#train model (to move the model from unknown parameters to known parameters)
#to know how poor the model is we use loss/cost/criterion function
#optimizer takes the handes onto loss and from that it tried to adjust the model parameter(the weight and bias)

#setup loss function
loss_fn=nn.L1Loss()#l1 is for mean abs error
#setup optimizer
optimizer= torch.optim.SGD(params=model_0.parameters(),
                           lr=0.01) #learning rate 
                           #SGD is a alogritm type build inside pytorch 

#training and testing loop
#steps- loop thorugh the data
        #forward pass (this involves data moving thorugh our models forward() function) to make prediction on data
        #caculate the loss (compare forward pass predcition to ground truth labels)
        #optimiser zero grad
        #loss backward - move backwards through the network to calculate the gradients of each of the parameter with repect to the loss
        #optimizer step- use the optimizer to adjsut our models parameter to improve the loss
#training loop
epochs= 100

for epoch in range(epochs):
    #set the model to training mode(its set by default)
    model_0.train()#sets all parameter  that requires gradient to require gradient
    #model.eval()#turns off the gradient tracking
    y_pred= model_0(x_train)
    #calculate the loss
    loss= loss_fn(y_pred,y_train)#prediction then labels calculate the mae betweeen them
    #print(f"loss:  {loss}")
    #optimizer zero grad
    optimizer.zero_grad()
    #performs backpropagation on the loss with respect to the parameters of the model and computed the weight (requires grad=true for that)
    loss.backward()
    #step the optimizer (perform gradient descent)
    optimizer.step() #by default how the optimizer changes will accumulate through the loop so.... we hvae to zero them above step 3 for the next iteration of loo p

#to save and load the model

#make model file and directory
model_path=Path("models")
model_path.mkdir(parents=True, exist_ok=True)

#create model save path
model_name= "model_1.pt"
model_save_path= model_path / model_name

torch.save(model_0.state_dict, model_save_path)#obj , save path. (state_dict is right apporch and more benificial in real world)


#print(model_0.state_dict())
#with torch.inference_mode():
   #y_pred_new=model_0(x_test)
#plot_prediction(prediction=y_pred_new)

#testing loop
model_0.eval()#turns of diffrent setting not needed during testing
with torch.inference_mode():#turns of gradient tracking
    #do the forward pass
    test_pred=model_0(x_test)
    #calculate the loss
    test_loss=loss_fn(test_pred,y_test)
    

print(f"epoch: {epoch} |  loss: {loss}| test loss: {test_loss}")
print(model_0.state_dict())
print(test_loss)
#plot_prediction(prediction=test_pred)#visualisation

