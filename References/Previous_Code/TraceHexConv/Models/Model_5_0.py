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

def Loss(y_pred, y_true,coeffs = [1,1,1,1]):
    assert y_pred[0].shape == y_true[0].shape, 'Energy shape is {} and {}'.format(y_pred[0].shape,y_true[0].shape)
    assert y_pred[1].shape == y_true[1].shape, 'Core shape is {} and {}'.format(y_pred[1].shape,y_true[1].shape)
    assert y_pred[2].shape == y_true[2].shape, 'Axis shape is {} and {}'.format(y_pred[2].shape,y_true[2].shape)
    assert y_pred[3].shape == y_true[3].shape, 'Xmax shape is {} and {}'.format(y_pred[3].shape,y_true[3].shape)
    criterion = nn.MSELoss()

    
    E_loss = coeffs[0]*criterion(y_pred[0],y_true[0])
    C_loss = coeffs[1]*criterion(y_pred[1],y_true[1])
    A_loss = coeffs[2]*criterion(y_pred[2],y_true[2])
    X_loss = coeffs[3]*criterion(y_pred[3],y_true[3])


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

def validate(model, dataloader_val, Loss_function, model_Coefficients,device):
    model.eval()
    val_T_loss = 0 
    val_E_loss = 0 
    val_C_loss = 0 
    val_A_loss = 0
    val_X_loss = 0

    with torch.no_grad():
        for batchD_main,batchD_aux, batchlogE,batchCore,batchAxis,batchXmax in dataloader_val:
            
            batchD_main = batchD_main.to(device)
            batchD_aux = batchD_aux.to(device)
            batchlogE = batchlogE.to(device)
            batchCore = batchCore.to(device)
            batchAxis = batchAxis.to(device)
            batchXmax = batchXmax.to(device)

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
        
        self.conv1 = hexagdly.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv2 =  hexagdly.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                hexagdly.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
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
        self.AveragePool = hexagdly.Conv2d_CustomKernel(sub_kernels = Hex_AvPool(in_channels=in_channels,out_channels = in_channels),stride = 2)
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

class New_Task_Block(nn.Module):
    def __init__(self, in_channels, size,out_channels):
        super(New_Task_Block, self).__init__()

        # input should be batch,features,11,11

        # Layers 
        self.Conv1 = nn.Conv2d(in_channels=in_channels, out_channels=size, kernel_size=3, stride=1, padding=1) # out = 11x11
        # Concatenate with Original stuff 
        self.Conv2 = nn.Conv2d(in_channels=in_channels+size, out_channels=size, kernel_size=3, stride=1, padding=1) # out = 11x11
        # Concatenate with Original stuff
        self.Conv3 = nn.Conv2d(in_channels=in_channels+size, out_channels=size, kernel_size=3, stride=1, padding=0) # out = 9x9
        # No more concatenate cause of shape difference
        self.Conv4 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=0) # out = 7x7
        self.Conv5 = nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding=0) # out = 5x5
        self.Conv6 = nn.Conv2d(in_channels=size, out_channels=1, kernel_size=1, stride=1, padding=0) # out = 5x5
        self.Flatten = nn.Flatten()
        self.FC1 = nn.Linear(25, 25)
        self.FC2 = nn.Linear(25, 25)
        self.FC3 = nn.Linear(25, out_channels)

    def forward(self,X):
        out = self.Conv1(X)
        out = torch.cat((out,X),dim=1)
        out = self.Conv2(out)
        out = torch.cat((out,X),dim=1)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = self.Conv6(out)
        out = self.Flatten(out)
        out = nn.LeakyReLU()(self.FC1(out))
        out = nn.LeakyReLU()(self.FC2(out))
        out = self.FC3(out)
        return out



    

        


class Trace_Block(nn.Module):
    def __init__(self, in_channels=3, hidden_dim = 20, features =12):
        super(Trace_Block, self).__init__()

        self.bi_lstm = nn.LSTM(input_size=in_channels, 
                                hidden_size=hidden_dim, 
                                batch_first=True, 
                                bidirectional=True)
        
        self.lstm = nn.LSTM(input_size=hidden_dim*2, # 2 for bidirection
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.CollectLSTM = nn.LSTM(input_size=hidden_dim,
                                    hidden_size=features,
                                    batch_first=True)
        self.features = features

    def forward(self,x):
        # Read parameters of input
        # print(x.shape)
        N,H,W,L,C = x.shape                 # Shape = N,H,W,L,C
        x = x.reshape(N*H*W,L,C)            # Shape = N*H*W,L,C

        out,_ = self.bi_lstm(x)             # Shape = N*H*W,L,hidden_dim*2
        out,_ = self.lstm(out)              # Shape = N*H*W,L,hidden_dim
        out,_ = self.CollectLSTM(out)       # Shape = N*H*W,L,features
        out = out[:,-1,:]                   # Shape = N*H*W,features (Only care about the final timestep here, aka "parameter")
        out = out.reshape(N,H,W,self.features) # Shape = N,H,W,features
        out = out.permute(0,3,1,2)          # Shape = N,features,H,W
        return out

    





class Model_5_0(nn.Module):

    def __init__(self,trace_features=12):
        super(Model_5_0, self).__init__()

        # Info
        self.Name = 'Model_5_0'
        self.Description ='''
        Adding the basic trace analysis to the (yet to be) working model from Jonas' Paper. 
        There are some differences,  but the main idea is the same.
        Jonas' uses two layers to analyze the trace, BD-LSTM  into LSTM.
        '''


        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,1,1,1]
        # self.LossCoefficients = [1,0,0,1]
        
        
        # Layers

        # One Trace analysis block

        self.Trace = Trace_Block(features = trace_features)

        # 3 Convolutional Layers + 3 concatenation layers
        # Task Blocks
        self.Energy = New_Task_Block(in_channels=trace_features+1, out_channels=1,size= 12)
        self.Core   = New_Task_Block(in_channels=trace_features+1, out_channels=2,size= 12)
        self.Axis   = New_Task_Block(in_channels=trace_features+1, out_channels=3,size= 12)
        self.Xmax   = New_Task_Block(in_channels=trace_features+1, out_channels=1,size= 12)

    def forward(self, traces, arrival):
            # Trace Analysis:
            
            traces = self.Trace(traces)

            # Concatenate the Aux data and the Trace data
            out = torch.cat((traces,arrival),dim=1) # Shape is now (N,features+1,H,W)
            

            # Task Blocks
            Energy = self.Energy(out)
            Core   = self.Core(out)
            Axis   = self.Axis(out)
            Xmax   = self.Xmax(out)
            # Core = torch.zeros(size = (Energy.shape[0],2)).to(Energy.device)
            # Axis = torch.zeros(size = (Energy.shape[0],3)).to(Energy.device)
            
            return Energy, Core, Axis, Xmax


