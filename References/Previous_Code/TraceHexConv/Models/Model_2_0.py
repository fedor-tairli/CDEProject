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
from torch.utils.checkpoint import checkpoint
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
        
        D_main = self.data[0][index]
        if D_main.shape[0]==1: D_main = D_main.squeeze(0)
        
        D_aux = self.data[1][index]
        if len(D_aux.shape)==2: D_aux = D_aux.unsqueeze(0)
        
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        y4 = self.targets[3][index]
        
        return D_main,D_aux, y1, y2, y3, y4


class MyDatasetForSparse(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.targets[0])
    def __getitem__(self, index):
        
        if self.data[0].is_sparse:
            # D_main = self.data[0].index_select(0,torch.tensor([index]).to(self.data[0].device))
            D_main = self.data[0].index_select(0,index)
            D_main = D_main.to_dense()
        else:
            D_main = self.data[0][index]
        
        if D_main.shape[0]==1: D_main = D_main.squeeze(0)
        
        D_aux = self.data[1][index]
        if len(D_aux.shape)==2: D_aux = D_aux.unsqueeze(0)
        
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        y4 = self.targets[3][index]
        
        return D_main,D_aux, y1, y2, y3, y4

def validate(model, dataloader_val, Loss_function, model_Coefficients,device='cuda'):
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

def Training_Track(model,dataloader,device = 'cuda'):
    Xmax_mean = 750
    Xmax_STD  = 66.80484050442804
    E_MEAN        = 19.0
    Norm_LEN      = 750.0
        
    logE_MAE = 0
    Core_MAE = 0
    Axis_MAE = 0
    Xmax_MAE = 0

    model.eval()
    for batchD_main,batchD_aux, logE_true,Core_true,Axis_true,Xmax_true in dataloader:
            
        batchD_main = batchD_main.to(device)
        batchD_aux = batchD_aux.to(device)
        logE_true = logE_true.to(device)
        Core_true = Core_true.to(device)
        Axis_true = Axis_true.to(device)
        Xmax_true = Xmax_true.to(device)

        with torch.no_grad():
            Results = model(batchD_main,batchD_aux)
        logE_pred = Results[0]
        Core_pred = Results[1]
        Axis_pred = Results[2]
        Xmax_pred = Results[3]
        
        # Unnormalize
        logE_pred = 10**(logE_pred+E_MEAN)-1
        logE_true = 10**(logE_true+E_MEAN)-1

        Core_pred = Core_pred*Norm_LEN
        Core_true = Core_true*Norm_LEN

        Axis_pred = F.normalize(Axis_pred, p=2, dim=1)  # make Axis_pred a unit vector
        # Axis_true should already be normalized

        Xmax_pred = Xmax_pred*Xmax_STD + Xmax_mean
        Xmax_true = Xmax_true*Xmax_STD + Xmax_mean

        # Calculate sum of absolute errors
        logE_MAE += torch.sum(torch.abs(logE_pred - logE_true))
        Core_MAE += torch.sum(torch.abs(Core_pred - Core_true))
        Axis_MAE += torch.sum(torch.abs(Axis_pred - Axis_true))
        Xmax_MAE += torch.sum(torch.abs(Xmax_pred - Xmax_true))

    # Calculate MAE
    logE_MAE /= len(dataloader.dataset)
    Core_MAE /= len(dataloader.dataset)
    Axis_MAE /= len(dataloader.dataset)
    Xmax_MAE /= len(dataloader.dataset)

    model.E_MSE_history.append(logE_MAE.item())
    model.C_MSE_history.append(Core_MAE.item())
    model.A_MSE_history.append(Axis_MAE.item())
    model.X_MSE_history.append(Xmax_MAE.item())

    track  = str('Unnormalised MAE ->'+' E: '+str(logE_MAE.item()  )[:9]+' C: '+str(Core_MAE.item()  )[:9]+' A: '+str(Axis_MAE.item()  )[:9]+' X: '+str(Xmax_MAE.item()  )[:9])
    return track


    


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


class Recurrent_Block(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, num_layers=1, dropout_rate=1, num_features=10):
        super(Recurrent_Block, self).__init__()

        # Bidirectional LSTM layer
        self.bidirectional_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)

        # Linear layer to transform output to desired number of features
        self.fc = nn.Linear(hidden_dim, num_features)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for name, param in self.bidirectional_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # input shape: (batch_size, sequence_length, num_channels, width, height)

        batch_size, sequence_length, num_channels, width, height = x.shape

        
        # rearrange input to shape: (batch_size*width*height, sequence_length, num_channels)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, sequence_length, num_channels)

        # pass data through Bidirectional LSTM layer
        bidir_lstm_out, _ = self.bidirectional_lstm(x)  # output shape: (batch_size*width*height, sequence_length, hidden_dim*2)

        # pass data through LSTM layers
        lstm_out, _ = self.lstm(bidir_lstm_out)  # output shape: (batch_size*width*height, sequence_length, hidden_dim)

        # apply linear layer to every time step
        features = self.fc(lstm_out[:, -1, :])  # output shape: (batch_size*width*height, num_features)

        # reshape features to original width and height, shape: (batch_size, height, width, num_features)
        features = features.view(batch_size, -1, width, height)

        return features



    


class Model_2_0(nn.Module):

    def __init__(self):
        super(Model_2_0, self).__init__()

        # Info
        self.Name = 'Model_2_0'
        self.Description ='''
        Having tested the memory usage of Hexagdly, the conclusion is that the hex-conv are not taking up too much space vs reg-conv.
        Hence, the dropping out of the task blocks will be attempted here. 

        The majority of memory is being used to track what happens in the TraceBlock. 
        I need to figure out how to reduce the memory usage of the TraceBlock.
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

        self.E_MSE_history = []
        self.C_MSE_history = []
        self.A_MSE_history = []
        self.X_MSE_history = []




        # self.LossCoefficients = [1300,1/3300,1/30,1/3000]
        self.LossCoefficients = [1,1,1,1]
        # self.LossCoefficients = [1,0,0,1]
        
        
        # Layers

        # One Trace analysis block

        self.Trace = Recurrent_Block()

        # 3 Convolutional Layers + 3 concatenation layers
        
        # in 3 channels
        self.Conv1 = hexagdly.Conv2d(in_channels=11, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 23 channels
        self.Conv2 = hexagdly.Conv2d(in_channels=23, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 35 channels
        self.Conv3 = hexagdly.Conv2d(in_channels=35, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 47 channels

        # Task Blocks
        self.Energy = Task_Block(in_channels=47, out_channels=1)
        self.Core   = Task_Block(in_channels=47, out_channels=2)
        self.Axis   = Task_Block(in_channels=47, out_channels=3)
        self.Xmax   = Task_Block(in_channels=47, out_channels=1)
    

    # Regular Forward

    def forward(self, traces, arrival):
        # Trace Analysis:
        out = self.Trace(traces)

        # Concatenate the Aux data and the Trace data
        out = torch.cat((out,arrival),dim=1) # Shape is now (N,11,H,W)
        
        # Conv Layers
        out = torch.cat((out,F.leaky_relu(self.Conv1(out))),dim=1)
        out = torch.cat((out,F.leaky_relu(self.Conv2(out))),dim=1)
        out = torch.cat((out,F.leaky_relu(self.Conv3(out))),dim=1)

        # Task Blocks
        Energy = self.Energy(out)
        Core   = self.Core(out)
        Axis   = self.Axis(out)
        Xmax   = self.Xmax(out)
        # Core = torch.zeros(size = (Energy.shape[0],2)).to(Energy.device)
        # Axis = torch.zeros(size = (Energy.shape[0],3)).to(Energy.device)
        
        return Energy, Core, Axis, Xmax



    # Checkpoint Forward
    # def _forward(self,out):
    #         # # Trace Analysis:
    #         # out = self.Trace(traces)

    #         # # Concatenate the Aux data and the Trace data
    #         # out = torch.cat((out,arrival),dim=1) # Shape is now (N,11,H,W)
            
    #         # Conv Layers
    #         out = torch.cat((out,F.relu(self.Conv1(out))),dim=1)
    #         out = torch.cat((out,F.relu(self.Conv2(out))),dim=1)
    #         out = torch.cat((out,F.relu(self.Conv3(out))),dim=1)

    #         # Task Blocks
    #         Energy = self.Energy(out)
    #         Core   = self.Core(out)
    #         Axis   = self.Axis(out)
    #         Xmax   = self.Xmax(out)
    #         # Core = torch.zeros(size = (Energy.shape[0],2)).to(Energy.device)
    #         # Axis = torch.zeros(size = (Energy.shape[0],3)).to(Energy.device)
            
    #         return Energy, Core, Axis, Xmax

    # def forward(self, traces, arrival):
    #     out = self.Trace(traces)
    #     out = torch.cat((out,arrival),dim=1) # Shape is now (N,11,H,W)
    #     return checkpoint(self._forward,out)
