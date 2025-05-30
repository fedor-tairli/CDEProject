# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import numpy as np
import hexagdly
import matplotlib.pyplot as plt
import os
import torch.nn.init as init


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

def UnnormaliseY(Y):
    # Unnorm -> out = np.log10(out+1)/np.log10(101)
    Y = Y*np.log10(101)
    Y = 10**Y
    Y = Y-1
    return Y

def UnnormaliseX(X):
    # Unnormalise Truth

    X[:,0] = X[:,0]+19
    X[:,1] = X[:,1]*66.08+750
    X[:,2] = X[:,2]
    X[:,3] = X[:,3]*750
    return(X)

def Loss(Pred,Truth):
    assert Pred.shape[1:] == torch.Size([120,1]), 'Pred shape is {}'.format(Pred.shape)
    assert Truth.shape[1:] == torch.Size([120,1]), 'Truth shape is {}'.format(Truth.shape)
    assert Pred.shape[0] == Truth.shape[0] , 'Batch Size doesnt Match o_0'
    criterion = nn.MSELoss()
    loss = criterion(Pred,Truth)
    return loss

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Main, Aux, Truth):
        self.Main = Main
        self.Aux  = Aux
        self.Truth = Truth
    def __len__(self):
        return len(self.Main)
    def __getitem__(self, idx):
        return self.Main[idx,:,:], self.Aux[idx,:], self.Truth[idx,:]
    
def validate(model,dataloader,Loss,device = 'cuda',Unnorm=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Main, Aux, Truth in dataloader:
            Main = Main.to(device)
            Aux  = Aux.to(device)
            Truth = Truth.to(device)
            Y_pred = model(Main,Aux)
            if Unnorm:
                Y_pred = model.UnnormaliseY(Y_pred)
                Truth  = model.UnnormaliseY(Truth)

            val_loss += Loss(Y_pred,Truth).item()
    return val_loss/len(dataloader)



class Model_2_squeeze(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_squeeze, self).__init__()
        # Info
        self.Name = 'Model_2_squeeze'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.squeeze().unsqueeze(1).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out


class Model_2_transpose12(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_transpose12, self).__init__()
        # Info
        self.Name = 'Model_2_transpose12'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.transpose(1,2).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out


class Model_2_transpose21(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_transpose21, self).__init__()
        # Info
        self.Name = 'Model_2_transpose21'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.transpose(2,1).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out


class Model_2_permute(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_permute, self).__init__()
        # Info
        self.Name = 'Model_2_permute'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.permute(0,2,1).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out

    
class Model_2_view(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_view, self).__init__()
        # Info
        self.Name = 'Model_2_view'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.view(-1,1,4).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out

  
class Model_2_reshape(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_2_reshape, self).__init__()
        # Info
        self.Name = 'Model_2_reshape'
        self.Description = 'Add the Auxiliary Data'
        # Layers
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        # Initialisation
        init_weights(self.Bi_LSTM)
        init_weights(self.LSTM)
        init_weights(self.CollectLSTM)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        seq_length = trace.size(1)

        # Tile auxiliary data across the time dimension and concatenate
        aux_tiled = aux.reshape(-1,1,4).repeat(1, seq_length, 1)
        
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = nn.ReLU()(out)

        if Unnorm:
            out = self.UnnormaliseY(out)
        
        return out

    



