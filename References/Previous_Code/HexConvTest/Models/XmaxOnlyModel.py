##############################################################
#                                                            #
#                  Here I define a simple model              #
#                   Train Exclusively Xmax                   #
#                                                            #
##############################################################





# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hexagdly




def Hex_AvPool(in_channels,out_channels):
    sub_kernels = [
                np.full((out_channels, in_channels, 3, 1), 1/7),
                np.full((out_channels, in_channels, 2, 2), 1/7)
                ]
    return sub_kernels

   
# Input shape is (N,3,11,11)

class Model_X_0(nn.Module):

    def __init__(self):
        super(Model_X_0, self).__init__()

        # Info
        self.Name = 'Model_X_0'
        self.Description ='''

        This simple model will only predict the Xmax of the shower.
        This here is to expore the way Xmax can be predicted
        Multiple iterations will likely be made to this model to find the best way to predict Xmax

        Trying SeLU activation on the FC layers 

        '''

        # History
        self.T_Loss_history = []

        self.T_Loss_history_val = []
        
        # Layers

        # 5 Convolutional Layers + 5 concatenation layers
        
        # in 3 channels
        self.Conv1 = hexagdly.Conv2d(in_channels=3, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 15 channels
        self.Conv2 = hexagdly.Conv2d(in_channels=15, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 27 channels
        self.Conv3 = hexagdly.Conv2d(in_channels=27, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 39 channels
        self.Conv4 = hexagdly.Conv2d(in_channels=39, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 51 channels
        self.Conv5 = hexagdly.Conv2d(in_channels=51, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 63 channels

        # 1 Average pooling
        self.AvgPool = hexagdly.Conv2d_CustomKernel(sub_kernels=Hex_AvPool(in_channels=63,out_channels=63),stride = 2)
        # This should bring down the sahpe to (N,3,5,6) (Or is it 6,5 - Doesnt matter)
        self.Conv6 = hexagdly.Conv2d(in_channels=63, out_channels=63, kernel_size=1,stride=1)
        # Possible Batch Normalisation Here, I think thats what might have screwed up the last model
        
        self.Conv7 = hexagdly.Conv2d(in_channels=63, out_channels=63, kernel_size=1,stride=1)
        
        # Some fully connected Layers : Will Test with 1,2,3, (Different Size Layers?)
        # Test with one layer
        self.FC1 = nn.Linear(in_features=63*5*6, out_features=100)
        self.FC2 = nn.Linear(in_features=100, out_features=100)
        self.FC3 = nn.Linear(in_features=100, out_features=100)
        self.Output = nn.Linear(in_features=100, out_features=1)



    def forward(self, x):
            # Conv Layers
            out = F.relu(self.Conv1(x))
            out = torch.cat((x,out),dim=1)
            out = torch.cat((out,F.relu(self.Conv2(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv3(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv4(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv5(out))),dim=1)
            out = self.AvgPool(out)
            out = F.relu(self.Conv6(out))
            out = F.relu(self.Conv7(out))
            # Flatten
            out = out.reshape(-1, 63*5*6)
            # FC Layers
            out = F.selu(self.FC1(out))
            out = F.selu(self.FC2(out))
            out = F.selu(self.FC3(out))
            # Output
            out = self.Output(out)


            return out



class Model_X_1(nn.Module):

    def __init__(self):
        super(Model_X_1, self).__init__()

        # Info
        self.Name = 'Model_X_1'
        self.Description ='''

        Something doesnt work, Fuck it , im going to just sploosh it with FCs and see it it works
        Tried basic Relu Activations
        Going to try TANH now. maybe need the negative sign, 
        Try Selu Next

        '''

        # History
        self.T_Loss_history = []

        self.T_Loss_history_val = []
        
        # Test with one layer
        self.FC1 = nn.Linear(in_features=3*11*11, out_features=256)
        self.FC2 = nn.Linear(in_features=256, out_features=256)
        self.FC3 = nn.Linear(in_features=256, out_features=256)
        self.FC4 = nn.Linear(in_features=256, out_features=256)
        self.FC5 = nn.Linear(in_features=256, out_features=256)
        self.FC6 = nn.Linear(in_features=256, out_features=128)
        self.FC7 = nn.Linear(in_features=128, out_features=64)
        self.FC8 = nn.Linear(in_features=64, out_features = 32)
        self.Output = nn.Linear(in_features=32, out_features=1)



    def forward(self, x):
            # Conv Layers
            # print(x.size())
            x = x.reshape((x.size()[0],3,11*11))
            # print(x.size())
            x = torch.cat((x[:,0],x[:,1],x[:,2]),axis =1)
            # print(x.size())
            out = F.tanh(self.FC1(x))
            out = F.tanh(self.FC2(out))
            out = F.tanh(self.FC3(out))
            out = F.tanh(self.FC4(out))
            out = F.tanh(self.FC5(out))
            out = F.tanh(self.FC6(out))
            out = F.tanh(self.FC7(out))
            out = F.tanh(self.FC8(out))
            # Output
            out = self.Output(out)


            return out


def Loss(y_pred, y_true):
    # For just Xmax, loss is a simple MSE
    return F.mse_loss(y_pred,y_true)

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,Xmax):
        self.data = data
        self.Xmax = Xmax
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[index,...]
        y = self.Xmax[index]
        return X, y

def validate(model, dataloader_val, Loss_function):
    model.eval()
    val_X_loss = 0 

    with torch.no_grad():
        for batchD, batchX in dataloader_val:
        
            predictions = model(batchD)
            
            X_loss = Loss_function(predictions,batchX)
            
            val_X_loss += X_loss.item()
            # break
    val_X_loss /= len(dataloader_val)

    return val_X_loss