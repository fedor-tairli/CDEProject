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

def Loss(y_pred, y_true):

    loss = torch.nn.functional.mse_loss(y_pred,y_true)
    return loss

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, index):
        
        if self.data[0].is_sparse:
            D_main = self.data[0].index_select(0,torch.tensor([index]).to(self.data[0].device))
        else:
            D_main = self.data[0][index]
        
        D_main = D_main.to_dense()
        if D_main.shape[0]==1: D_main = D_main.squeeze(0)
        
        D_aux = self.data[1][index]
        if len(D_aux.shape)==2: D_aux = D_aux.unsqueeze(0)
        
        y1 = self.targets[index]
        
        return D_main,D_aux, y1

def validate(model, dataloader_val, Loss_function,device='cuda'):
    model.eval()
    val_X_loss = 0

    with torch.no_grad():
        for batchD_main,batchD_aux,batchXmax in dataloader_val:
            batchD_main = batchD_main.to(device)
            batchD_aux = batchD_aux.to(device)
            batchXmax = batchXmax.to(device)

            predictions = model(batchD_main,batchD_aux)
            
            X_loss = Loss_function(predictions,batchXmax)
            
            val_X_loss += X_loss.item()
            # break
    val_X_loss /= len(dataloader_val)

    return val_X_loss


# Hexagdly average pool in Hex Regime
def Hex_AvPool(in_channels,out_channels):
    sub_kernels = [
                np.full((out_channels, in_channels, 3, 1), 1/7),
                np.full((out_channels, in_channels, 2, 2), 1/7)
                ]
    return sub_kernels

# Custom definition for residual block, to be used with hexagdly

# Define Custom Task Block one for each task

class Recurrent_Block(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=10, num_layers=1, dropout_rate=1, num_features=10):
        super(Recurrent_Block, self).__init__()

        # Bidirectional LSTM layer
        self.bidirectional_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim*2, num_features, num_layers, batch_first=True, dropout=dropout_rate)

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

    def forward(self, x):
        # input shape: (batch_size, sequence_length, num_channels, width, height)

        batch_size, sequence_length, num_channels, width, height = x.shape

        # rearrange input to shape: (batch_size*width*height, sequence_length, num_channels)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, sequence_length, num_channels)

        # pass data through Bidirectional LSTM layer
        bidir_lstm_out, _ = self.bidirectional_lstm(x)  # output shape: (batch_size*width*height, sequence_length, hidden_dim*2)

        # pass data through LSTM layers
        lstm_out, _ = self.lstm(bidir_lstm_out)  # output shape: (batch_size*width*height, sequence_length, hidden_dim)

        
        features = lstm_out[:, -1, :]  # output shape: (batch_size*width*height, num_features)

        # reshape features to original width and height, shape: (batch_size, height, width, num_features)
        features = features.view(batch_size, -1, width, height)

        return features



    


class Model_X_0(nn.Module):

    def __init__(self):
        super(Model_X_0, self).__init__()

        # Info
        self.Name = 'Model_X_0'
        self.Description ='''
        Testing Why the heck Xmax doesn't train at all.
        Very confusing
        This model will only predict Xmax.
        '''


        # History
        self.X_Loss_history = []

        self.X_Loss_history_val = []

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
        self.Conv4 = hexagdly.Conv2d(in_channels=47, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 59 channels
        self.Conv5 = hexagdly.Conv2d(in_channels=59, out_channels=12, kernel_size=1,stride=1)
        # Concatenate -> in 71 channels
        self.AvPool = hexagdly.Conv2d_CustomKernel(sub_kernels = Hex_AvPool(in_channels=71,out_channels = 71),stride = 2)

        self.Conv6 = hexagdly.Conv2d(in_channels=71, out_channels=12, kernel_size=1,stride=1)

        self.GlobalPool = nn.AdaptiveAvgPool2d(1)

        # some fully connected layers

        self.FC1 = nn.Linear(12, 12)
        self.FC2 = nn.Linear(12, 12)
        self.FC3 = nn.Linear(12, 1)
        

    # Regular Forward

    def forward(self, traces, arrival):
        # Trace Analysis:
        out = self.Trace(traces)

        # Concatenate the Aux data and the Trace data
        out = torch.cat((out,arrival),dim=1) # Shape is now (N,11,H,W)
        
        # Conv Layers
        out = torch.cat((out,F.selu(self.Conv1(out))),dim=1)
        out = torch.cat((out,F.selu(self.Conv2(out))),dim=1)
        out = torch.cat((out,F.selu(self.Conv3(out))),dim=1)
        out = torch.cat((out,F.selu(self.Conv4(out))),dim=1)
        out = torch.cat((out,F.selu(self.Conv5(out))),dim=1)

        out = self.AvPool(out)

        out = F.selu(self.Conv6(out))

        # Reshape for FC layers
        out = self.GlobalPool(out)
        out = out.view(out.size(0), -1)

        # FC layers

        out = F.selu(self.FC1(out))
        out = F.selu(self.FC2(out))
        out = self.FC3(out)
                
        return out


