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


def Loss(Pred,Truth):
    assert Pred.shape[1:] == torch.Size([1]), 'Pred shape is {}'.format(Pred.shape)      # Already Summed
    if Truth.shape[1:] == torch.Size([120,1]):
        Truth = Truth.sum(dim=1)                                                           # Not Already Summed
    else: 
        assert Truth.shape[1:] == torch.Size([1]), 'Truth shape is {}'.format(Truth.shape) # Already Summed
    assert Pred.shape == Truth.shape , 'Shapes dont Match'
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    loss = criterion(Pred,Truth)
    return loss

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

def validate(model,dataloader,Loss,device = 'cuda',Unnorm=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Main, Aux, Truth in dataloader:
            Main = Main.to(device)
            Aux  = Aux.to(device)
            Truth = Truth.to(device)
            Y_pred,_ = model(Main,Aux)
            if Unnorm:
                Y_pred = model.UnnormaliseY(Y_pred)
                Truth  = model.UnnormaliseY(Truth)

            val_loss += Loss(Y_pred,Truth).item()
    return val_loss/len(dataloader)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Main, Aux, Truth):
        self.Main = Main
        self.Aux  = Aux
        self.Truth = Truth
    def __len__(self):
        return len(self.Main)
    def __getitem__(self, idx):
        return self.Main[idx,:,:], self.Aux[idx,:], self.Truth[idx,:]
    



class Model_3_0(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4,features=12):
        super(Model_3_0, self).__init__()
        # Info
        self.Name = 'Model_3_0'
        self.Description = '''
        Try to produce the total useing the Summation instead of features
        Use S_mu / S_tot as the Truth
        '''
        # Layers
        self.Bi_LSTM     = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM        = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
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
        out    = self.UnnormaliseY(out).sum(dim=1)/self.UnnormaliseY(trace).mean(dim = 2).sum(dim=1).unsqueeze(1)

        if Unnorm:
            print('Unnormalisation impossible on integrated signal')
        
        return out


    



    
class Model_4_Features(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4,features=12):
        super(Model_4_Features, self).__init__()
        # Info
        self.Name = 'Model_4_Features'
        self.Description = '''
        Try to produce the total Signal Using 12 features, use only 2 FC layers first
        Use the S_mu / S_tot as the Truth
        Test what is more important for the model Xmax, logE, CosZen or Core Distance
        '''

        # Layers
        self.Bi_LSTM     = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM        = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, features, batch_first=True)
        self.FC          = nn.Linear(features, hidden_dim)
        self.FC2         = nn.Linear(hidden_dim, 1)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        # Zero appropriate aux variables
        # order [logE,Xmax,CosZen,CoreDist]
        # Tile auxiliary data across the time dimension and concatenate
        seq_length = trace.size(1)
        aux_tiled = aux.transpose(1,2).repeat(1, seq_length, 1)
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)
        

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = out[:,-1,:]
        features = out
        out    = nn.LeakyReLU()(self.FC(out))
        out    = nn.Sigmoid()(self.FC2(out))

        return out, features

class Model_4_Features2(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4,features=12):
        super(Model_4_Features2, self).__init__()
        # Info
        self.Name = 'Model_4_Features2'
        self.Description = '''
        Try to produce the total Signal Using 12 features, use only 2 FC layers first
        Use the S_mu / S_tot as the Truth
        Test what is more important for the model Xmax, logE, CosZen or Core Distance
        '''

        # Layers
        self.Bi_LSTM     = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM        = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, features, batch_first=True)
        self.FC          = nn.Linear(features, hidden_dim)
        self.FC2         = nn.Linear(hidden_dim, 1)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        # Zero appropriate aux variables
        # order [logE,Xmax,CosZen,CoreDist]
        # Tile auxiliary data across the time dimension and concatenate
        seq_length = trace.size(1)
        aux_tiled = aux.transpose(1,2).repeat(1, seq_length, 1)
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)
        

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = out[:,-1,:]
        features = out
        out    = nn.LeakyReLU()(self.FC(out))
        out    = nn.Sigmoid()(self.FC2(out))

        return out, features

class Model_4_Features3(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4,features=3):
        super(Model_4_Features3, self).__init__()
        # Info
        self.Name = 'Model_4_Features3'
        self.Description = '''
        Try to produce the total Signal Using 12 features, use only 2 FC layers first
        Use the S_mu / S_tot as the Truth
        Test what is more important for the model Xmax, logE, CosZen or Core Distance
        '''

        # Layers
        self.Bi_LSTM     = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM        = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, features, batch_first=True)
        self.FC          = nn.Linear(features, hidden_dim)
        self.FC2         = nn.Linear(hidden_dim, 1)
        # Functions
        self.UnnormaliseY = UnnormaliseY
        self.UnnormaliseX = UnnormaliseX

    def forward(self, trace, aux, Unnorm=False):
        # Zero appropriate aux variables
        # order [logE,Xmax,CosZen,CoreDist]
        # Tile auxiliary data across the time dimension and concatenate
        seq_length = trace.size(1)
        aux_tiled = aux.transpose(1,2).repeat(1, seq_length, 1)
        trace_with_aux = torch.cat((trace, aux_tiled), dim=2)
        

        out, _ = self.Bi_LSTM(trace_with_aux)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        out    = out[:,-1,:]
        features = out
        out    = nn.LeakyReLU()(self.FC(out))
        out    = nn.Sigmoid()(self.FC2(out))

        return out, features
