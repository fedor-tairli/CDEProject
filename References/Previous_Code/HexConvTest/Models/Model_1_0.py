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
        self.AveragePool = hexagdly.Conv2d_CustomKernel(sub_kernels = Hex_AvPool(in_channels=39,out_channels = 39),stride = 2)
        self.ResBlock2 = ResidualBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        # Global AvPool
        self.GlobAvPool = nn.AdaptiveAvgPool2d(1)
        # outchannels = in_channels
        # 1 FC layer
        self.FC = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.ResBlock1(x)
        out = self.AveragePool(out)
        out = self.ResBlock2(out)
        out = self.GlobAvPool(out)
        out = out.reshape(-1,39)
        out = self.FC(out)
        
        return out


        

# Input shape is (N,3,11,11)





class Model_1_0(nn.Module):

    def __init__(self):
        super(Model_1_0, self).__init__()

        # Info
        self.Name = 'Model_1_0'
        self.Description ='''
        Define a model as closely as possible to what Jonas did 
        Obviously Omitting the reccurent part and going straight into convolutions       
        Its a bit more complex than what i thought it was.
        Going to just Yolo it with a simple model and improve in next iterations if possible.
        Need to get the reccurent part also, (later)
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

        # 3 Convolutional Layers + 3 concatenation layers
        
        # in 3 channels
        self.Conv1 = hexagdly.Conv2d(in_channels=3, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 15 channels
        self.Conv2 = hexagdly.Conv2d(in_channels=15, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 27 channels
        self.Conv3 = hexagdly.Conv2d(in_channels=27, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 39 channels

        # Task Blocks
        self.Energy = Task_Block(in_channels=39, out_channels=1)
        self.Core   = Task_Block(in_channels=39, out_channels=2)
        self.Axis   = Task_Block(in_channels=39, out_channels=3)
        self.Xmax   = Task_Block(in_channels=39, out_channels=1)

    def forward(self, x):
            # Conv Layers
            out = F.relu(self.Conv1(x))
            out = torch.cat((x,out),dim=1)
            out = torch.cat((out,F.relu(self.Conv2(out))),dim=1)
            out = torch.cat((out,F.relu(self.Conv3(out))),dim=1)

            # Task Blocks
            Energy = self.Energy(out)
            Core   = self.Core(out)
            Axis   = self.Axis(out)
            Xmax   = self.Xmax(out)

            return Energy, Core, Axis, Xmax




def Loss(y_pred, y_true,coeffs):
    # Loss is the sum of mean square error for each output scaled by coeffs
    # print('y_pred',y_pred)
    # print('y_true',y_true)
    # Print shape of X_max in both y_true and y_pred

    E_loss = coeffs[0]*torch.mean((y_pred[0] - y_true[0])**2) # Need to squeeze the pred?
    C_loss = coeffs[1]*torch.mean((y_pred[1] - y_true[1])**2)
    A_loss = coeffs[2]*torch.mean((y_pred[2] - y_true[2])**2)
    X_loss = coeffs[3]*torch.mean((y_pred[3] - y_true[3])**2) # Need to squeeze the pred? 
    
    # print()
    # print('y_pred[3].shape',y_pred[3].squeeze().shape)
    # print('y_true[3].shape',y_true[3].shape)
    # print('X_loss',X_loss)
    

    T_Loss = E_loss + C_loss + A_loss+ X_loss
    
    return T_Loss, E_loss, C_loss, A_loss, X_loss

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        X = self.data[index,...]
        y1 = self.targets[0][index]
        y2 = self.targets[1][index]
        y3 = self.targets[2][index]
        y4 = self.targets[3][index]
        return X, y1, y2, y3, y4

def validate(model, dataloader_val, Loss_function, model_Coefficients):
    model.eval()
    val_T_loss = 0 
    val_E_loss = 0 
    val_C_loss = 0 
    val_A_loss = 0
    val_X_loss = 0

    with torch.no_grad():
        for batchD, batchlogE,batchcore,batchaxis,batchX in dataloader_val:
        
            predictions = model(batchD)
            
            T_loss,E_loss,C_loss,A_loss,X_loss = Loss_function(predictions,(batchlogE,batchcore,batchaxis,batchX),coeffs = model_Coefficients)
            
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