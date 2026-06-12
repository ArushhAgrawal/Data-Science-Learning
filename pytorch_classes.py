#pytorch model building essentials

#torch.nn - contains all the building block for computaional grpah (a nural network can be considered a computaional graph).
#torch.nn.Parameter - what parameter our model shuld try and learn, often a Pytorch layer from torch.nn will set these for us.
#torch.nn.Module - the base class for all neural network modules, if you subclass it, you should write forward.
#torch.optim - this is where the optimizers in pytorch live, they will help with gradient descent IT MAKES SURE THAT THE VALUES THAT ARE IN OUR DATA IS NOT RANDOM INSTED ITS THE VALUE THAT REPRESENTS OUR DATA CLOSELY.
#def forward() - all nn.Module subclass requires u to overite forward, depending what u want ur model to do it becomes more complex.
