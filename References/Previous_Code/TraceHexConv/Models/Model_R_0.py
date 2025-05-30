##############################################################
#                 Here we define the models                  #
#               And custom torch datastructures              #
#            We iterate with simple version numbers          #
##############################################################


# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hexagdly

# Define the custom Datastructures and loss and validation functions

def Loss(y_pred, y_true,coeffs):

    E_loss = coeffs[0]*torch.nn.functional.mse_loss(y_pred[0],y_true[0])
    C_loss = coeffs[1]*torch.nn.functional.mse_loss(y_pred[1],y_true[1])
    A_loss = coeffs[2]*torch.nn.functional.mse_loss(y_pred[2],y_true[2])
    X_loss = coeffs[3]*torch.nn.functional.mse_loss(y_pred[3],y_true[3])

    T_Loss = E_loss + C_loss + A_loss+ X_loss

    return T_Loss, E_loss, C_loss, A_loss, X_loss

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.targets[0])
    def __getitem__(self, index):
        
        if self.data[0].is_sparse:
            D_main = self.data[0].index_select(0,torch.tensor([index]).to(self.data[0].device))
        else:
            D_main = self.data[0][index]
        
        D_main = D_main.to_dense()
        if D_main.shape[0]==1: D_main = D_main.squeeze(0)
        
        D_aux = self.data[1][index]
        if len(D_aux.shape)==2: D_aux = D_aux.unsqueeze(0)
        
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        y4 = self.targets[3][index]
        
        return D_main,D_aux, y1, y2, y3, y4

def validate(model, dataloader_val, Loss_function, model_Coefficients):
    model.eval()
    val_T_loss = 0 
    val_E_loss = 0 
    val_C_loss = 0 
    val_A_loss = 0
    val_X_loss = 0

    with torch.no_grad():
        for batchD_main,batchD_aux, batchlogE,batchCore,batchAxis,batchXmax in dataloader_val:
        
            predictions = model(batchD_main,batchD_aux)
            
            T_loss,E_loss,C_loss,A_loss,X_loss = Loss_function(predictions,(batchlogE,batchCore,batchAxis,batchXmax),coeffs = model_Coefficients)
            
            val_T_loss += T_loss.item()
            val_E_loss += E_loss.item()
            val_C_loss += C_loss.item()
            val_A_loss += A_loss.item()
            val_X_loss += X_loss.item()
            # break
    val_T_loss /= len(dataloader_val)
    val_E_loss /= len(dataloader_val)
    val_C_loss /= len(dataloader_val)
    val_A_loss /= len(dataloader_val)
    val_X_loss /= len(dataloader_val)

    return val_T_loss, val_E_loss, val_C_loss, val_A_loss, val_X_loss




# Hexagdly average pool in Hex Regime
def Hex_AvPool(in_channels,out_channels):
    sub_kernels = [
                np.full((out_channels, in_channels, 3, 1), 1/7),
                np.full((out_channels, in_channels, 2, 2), 1/7)
                ]
    return sub_kernels

# Custom definition for residual block, to be used with hexagdly
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 =  nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # If the number of input channels and output channels are not equal
        # we need to transform the residual (input) tensor to have the same
        # number of channels as the output
        residual = self.skip(residual)

        out += residual
        out = self.relu(out)

        return out
  


# Define Custom Task Block one for each task
class Task_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Task_Block, self).__init__()

        # 2 Residual Blocks 
        self.ResBlock1 = ResidualBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        # average pooling
        self.AveragePool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.ResBlock2 = ResidualBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        # Global AvPool
        self.GlobAvPool = nn.AdaptiveAvgPool2d(1)
        # outchannels = in_channels
        # 1 FC layer
        self.FC = nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
    def forward(self, x):
        out = self.ResBlock1(x)
        out = self.AveragePool(out)
        out = self.ResBlock2(out)
        out = self.GlobAvPool(out)
        out = out.reshape(-1,self.in_channels)
        out = self.FC(out)
        
        return out


class Trace_Block(nn.Module):
    def __init__(self):
        super(Trace_Block, self).__init__()

        self.bi_lstm = nn.LSTM(input_size=3, 
                                hidden_size=50, 
                                num_layers=2, 
                                batch_first=True, 
                                bidirectional=True)
        
        self.lstm = nn.LSTM(input_size=100, # 2 for bidirection
                            hidden_size=10,
                            num_layers=2,
                            batch_first=True)
        
    def forward(self,x):
        # Read parameters of input
        # print(x.shape)
        N,L,C,H,W = x.shape
        # Prepare output
        output = torch.zeros(N,H,W,10).to(x.device)

        # Process each tank individually:
        for i in range(H):
            for j in range(W):
                trace = x[:,:,:,i,j]        # Shape = N,120,3

                out,_ = self.bi_lstm(trace) # Shape = N,120,100
                out,_ = self.lstm(out)      # Shape = N,120,10
                out = out[:,-1,:]           # Shape = N,10 (Only care about the findal timestep here, aka "parameter")
                output[:,i,j,:] = out     
        output = output.permute(0,3,1,2)    # Shape = N,10,H,W
        return output
    





class Model_R_0(nn.Module):

    def __init__(self):
        super(Model_R_0, self).__init__()

        # Info
        self.Name = 'Model_R_0'
        self.Description ='''
        Testing the effect of using Regular convolutional layers instead of hexagonal ones
        Test to see the memory usage, if it doesn't improve, Than we can continue with hexagonal layers
        '''


        # History
        self.T_Loss_history = []
        self.E_Loss_history = []
        self.X_Loss_history = []
        self.C_Loss_history = []
        self.A_Loss_history = []

        self.T_Loss_history_val = []
        self.E_Loss_history_val = []
        self.X_Loss_history_val = []
        self.C_Loss_history_val = []
        self.A_Loss_history_val = []

        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,1,1,1]
        # self.LossCoefficients = [1,0,0,1]
        
        
        # Layers

        # One Trace analysis block

        self.Trace = Trace_Block()

        # 3 Convolutional Layers + 3 concatenation layers
        
        # in 3 channels
        self.Conv1 = nn.Conv2d(in_channels=11, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 23 channels
        self.Conv2 = nn.Conv2d(in_channels=23, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 35 channels
        self.Conv3 = nn.Conv2d(in_channels=35, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 47 channels

        # Task Blocks
        self.Energy = Task_Block(in_channels=47, out_channels=1)
        self.Core   = Task_Block(in_channels=47, out_channels=2)
        self.Axis   = Task_Block(in_channels=47, out_channels=3)
        self.Xmax   = Task_Block(in_channels=47, out_channels=1)

    def forward(self, traces, arrival):
            # Trace Analysis:
            out = self.Trace(traces)

            # Concatenate the Aux data and the Trace data
            out = torch.cat((out,arrival),dim=1) # Shape is now (N,11,H,W)
            
            # Conv Layers
            out = torch.cat((out,F.relu(self.Conv1(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv2(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv3(out))),dim=1)

            # Task Blocks
            Energy = self.Energy(out)
            Core   = self.Core(out)
            Axis   = self.Axis(out)
            Xmax   = self.Xmax(out)
            # Core = torch.zeros(size = (Energy.shape[0],2)).to(Energy.device)
            # Axis = torch.zeros(size = (Energy.shape[0],3)).to(Energy.device)
            
            return Energy, Core, Axis, Xmax


