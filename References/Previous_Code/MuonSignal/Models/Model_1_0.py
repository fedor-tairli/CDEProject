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



class Model_1_0(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=50, output_dim=1):
        super(Model_1_0, self).__init__()
        # Info
        self.Name = 'Model_1_0'
        self.Description ='''
        First Iteration at predicting the muon signal.
        Simply Look at the integral of the signal. 
        '''
        
        # Bidirectional LSTM for 3-channel signal analysis
        self.bi_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # LSTM for signal production
        self.lstm1 = nn.LSTM(2 * hidden_dim, hidden_dim, batch_first=True)  
        # LSTM for signal integration
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Fully connected layer to produce a single value prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

        
        
    def forward(self, trace,aux,Unnorm=False):
        # Pass through bidirectional LSTM
        out, _ = self.bi_lstm(trace)
        
        # Pass through first LSTM
        out, _ = self.lstm1(out)
        
        # Pass through second LSTM
        out, _ = self.lstm2(out)
        
        out = out.sum(dim=1)
        
        out = self.fc(out)
        
        
        if Unnorm: out = self.UnnormaliseY(out)
        
        return out



    def UnnormaliseY(self, Y):
        # Unnorm -> out = np.log10(out+1)/np.log10(101)
        Y = Y*np.log10(101)
        Y = 10**Y
        Y = Y-1
        return Y
    
    def UnnormaliseX(self, X):
        # Unnormalise Truth

        X[:,0] = X[:,0]+19
        X[:,1] = X[:,1]*66.08+750
        X[:,2] = X[:,2]
        X[:,3] = X[:,3]*750
        return(X)
    

    



class Model_1_1(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=1, output_dim=1):
        super(Model_1_1, self).__init__()
        # Info
        self.Name = 'Model_1_1'
        self.Description ='''
        No Nothing, OONGABOONGA TIME
        '''
        
        self.Param = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.Bias  = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        
    def forward(self, trace,aux,Unnorm=False):
        
        trace = trace.mean(dim=2)
        trace = trace * self.Param
        trace = trace + self.Bias

        out = trace.sum(dim=1)
        
        if Unnorm: out = self.UnnormaliseY(out)
        
        return out



    def UnnormaliseY(self, Y):
        # Unnorm -> out = np.log10(out+1)/np.log10(101)
        Y = Y*np.log10(101)
        Y = 10**Y
        Y = Y-1
        return Y
    
    def UnnormaliseX(self, X):
        # Unnormalise Truth

        X[:,0] = X[:,0]+19
        X[:,1] = X[:,1]*66.08+750
        X[:,2] = X[:,2]
        X[:,3] = X[:,3]*750
        return(X)
    

    



class Model_1_2(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, output_dim=1):
        super(Model_1_2, self).__init__()
        # Info
        self.Name = 'Model_1_2'
        self.Description ='''
        One Layer
        '''
        self.LSTM        = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        
    def forward(self, trace,aux,Unnorm=False):
        
        out, _ = self.LSTM(trace)
        out, _ = self.CollectLSTM(out)
        

        if Unnorm: out = self.UnnormaliseY(out)
        
        return out



    def UnnormaliseY(self, Y):
        # Unnorm -> out = np.log10(out+1)/np.log10(101)
        Y = Y*np.log10(101)
        Y = 10**Y
        Y = Y-1
        return Y
    
    def UnnormaliseX(self, X):
        # Unnormalise Truth

        X[:,0] = X[:,0]+19
        X[:,1] = X[:,1]*66.08+750
        X[:,2] = X[:,2]
        X[:,3] = X[:,3]*750
        return(X)


class Model_1_3(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, output_dim=1):
        super(Model_1_3, self).__init__()
        # Info
        self.Name = 'Model_1_3'
        self.Description ='''
        UpGrade Time
        '''
        self.Bi_LSTM     = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM        = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)
        
    def forward(self, trace,aux,Unnorm=False):

        out, _ = self.Bi_LSTM(trace)
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        

        if Unnorm: out = self.UnnormaliseY(out)
        
        return out



    def UnnormaliseY(self, Y):
        # Unnorm -> out = np.log10(out+1)/np.log10(101)
        Y = Y*np.log10(101)
        Y = 10**Y
        Y = Y-1
        return Y
    
    def UnnormaliseX(self, X):
        # Unnormalise Truth

        X[:,0] = X[:,0]+19
        X[:,1] = X[:,1]*66.08+750
        X[:,2] = X[:,2]
        X[:,3] = X[:,3]*750
        return(X)


class Model_1_4(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=20, aux_dim=4):
        super(Model_1_4, self).__init__()
        self.Name = 'Model_1_4'
        self.Description = 'Add the Auxiliary Data'
        
        self.Bi_LSTM = nn.LSTM(input_dim + aux_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim, 1, batch_first=True)

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


    def UnnormaliseY(self, Y):
        # Unnorm -> out = np.log10(out+1)/np.log10(101)
        Y = Y*np.log10(101)
        Y = 10**Y
        Y = Y-1
        return Y
    
    def UnnormaliseX(self, X):
        # Unnormalise Truth

        X[:,0] = X[:,0]+19
        X[:,1] = X[:,1]*66.08+750
        X[:,2] = X[:,2]
        X[:,3] = X[:,3]*750
        return(X)