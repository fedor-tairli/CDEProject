# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
# from time import time
# from   torch_geometric.nn import GCNConv, TAGConv,GATConv
# from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
# from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['x','y','z','SDPPhi','CEDist'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
    
    losses['Total'] = sum(losses.values())
    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses


def validate(model,Dataset,Loss,device,BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth,_  in Dataset:
            
            Preds .append( model(BatchMains,BatchAux).to('cpu'))
            Truths.append(       BatchTruth          .to('cpu'))
        Preds  = torch.cat(Preds ,dim=0)
        Truths = torch.cat(Truths,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss(Preds,Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)
    

def metric(model,Dataset,device,keys = ['x','y','z','SDPPhi','CEDist'],BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the 68% containment range of the angular deviation
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = BatchSize
    Preds  = []
    Truths = []
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth, _ in Dataset:
            Preds .append( model(BatchMains,BatchAux).to('cpu'))
            Truths.append(       BatchTruth          .to('cpu'))
    Preds  = torch.cat(Preds ,dim=0).cpu()
    Truths = torch.cat(Truths,dim=0).cpu()
    Preds  = Dataset.Unnormalise_Truth(Preds )
    Truths = Dataset.Unnormalise_Truth(Truths)

    Units = Dataset.Truth_Units
    metrics = {}
    for i,key in enumerate(keys):
        if Units[i] == 'rad':
            AngDiv = torch.atan2(torch.sin(Preds[:,i]-Truths[:,i]),torch.cos(Preds[:,i]-Truths[:,i]))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)
        if Units[i] == 'deg':
            AngDiv = torch.atan2(torch.sin(torch.deg2rad(Preds[:,i]-Truths[:,i])),torch.cos(torch.deg2rad(Preds[:,i]-Truths[:,i])))
            metrics[key] = torch.quantile(torch.abs(AngDiv),0.68)*180/torch.pi
        else:
            metrics[key] = torch.quantile(torch.abs(Preds[:,i]-Truths[:,i]),0.68)
    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return metrics

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x



class Model_Axis_LSTM(nn.Module):
    Name = 'Model_Axis_LSTM'
    Description = '''
    LSTM Architecture for Axis Reconstruction
    Uses standad LSTM layers
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 2, 'Only one Main Channel is expected'
        in_station_channels = in_main_channels[1] # a bit of a hack, to reassign main channels into the same variable
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        
        super(Model_Axis_LSTM, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()

        # LSTM Layers
        # First bidirectional with n layers and dropout
        self.LSTM_main = nn.LSTM(in_main_channels, N_LSTM_nodes, num_layers=N_LSTM_layers, batch_first=True, dropout=dropout, bidirectional=True)
        # Second unidirection with 1 layer to get the output
        self.LSTM_summ = nn.LSTM(N_LSTM_nodes*2, N_LSTM_nodes, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)

        # Cut the last layer output to get the trace features

        # Aux Layers to add station data
        self.Dense1 = nn.Linear(in_station_channels+N_LSTM_nodes, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.X1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.X2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.X3 = nn.Linear(N_dense_nodes//2,1)

        self.Y1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Y2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Y3 = nn.Linear(N_dense_nodes//2,1)

        self.Z1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Z2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Z3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

        self.CEDist1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.CEDist2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.CEDist3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1,1,1,1])

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 2, 'Expected 2 mains'
        Station = Main[1].to(device) # a bit of a hack, to reassign main channels into the same variable
        Main    = Main[0].to(device)

        # Check if Main or Aux have any nan values
        # LSTM Layers
        Main, _ = self.LSTM_main(Main)
        Main, _ = self.LSTM_summ(Main)

        Main = Main[:,-1,:] # Take the last output as features
        # Station Data
        Main = torch.cat([Station,Main],dim=1)
        # Aux Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        # Dense and Output Layers
        X = self.Dense_Activation(self.X1(Main))
        X = self.Dense_Activation(self.X2(X))
        X = self.X3(X)

        Y = self.Dense_Activation(self.Y1(Main))
        Y = self.Dense_Activation(self.Y2(Y))
        Y = self.Y3(Y)

        Z = self.Dense_Activation(self.Z1(Main))
        Z = self.Dense_Activation(self.Z2(Z))
        Z = self.Z3(Z)
        
        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        CEDist = self.Dense_Activation(self.CEDist1(Main))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)


        Output = torch.cat([X,Y,Z,Phi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output


class Model_Axis_LSTM_JustX(Model_Axis_LSTM):
    Name = 'Model_Axis_LSTM_JustX'
    Description = '''
    LSTM Neural Network for SDP Reconstruction
    Only the X is learned
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        super(Model_Axis_LSTM_JustX, self).__init__(in_main_channels=in_main_channels, N_LSTM_nodes=N_LSTM_nodes, N_LSTM_layers=N_LSTM_layers, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1,0,0,0,0])

class Model_Axis_LSTM_JustY(Model_Axis_LSTM):
    Name = 'Model_Axis_LSTM_JustY'
    Description = '''
    LSTM Neural Network for SDP Reconstruction
    Only the Y is learned
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        super(Model_Axis_LSTM_JustY, self).__init__(in_main_channels=in_main_channels, N_LSTM_nodes=N_LSTM_nodes, N_LSTM_layers=N_LSTM_layers, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,1,0,0,0])

class Model_Axis_LSTM_JustZ(Model_Axis_LSTM):
    Name = 'Model_Axis_LSTM_JustZ'
    Description = '''
    LSTM Neural Network for SDP Reconstruction
    Only the Z is learned
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        super(Model_Axis_LSTM_JustZ, self).__init__(in_main_channels=in_main_channels, N_LSTM_nodes=N_LSTM_nodes, N_LSTM_layers=N_LSTM_layers, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,1,0,0])


class Model_Axis_LSTM_JustSDPPhi(Model_Axis_LSTM):
    Name = 'Model_Axis_LSTM_JustSDPPhi'
    Description = '''
    LSTM Neural Network for SDP Reconstruction
    Only the SDPPhi is learned
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        super(Model_Axis_LSTM_JustSDPPhi, self).__init__(in_main_channels=in_main_channels, N_LSTM_nodes=N_LSTM_nodes, N_LSTM_layers=N_LSTM_layers, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,1,0])


class Model_Axis_LSTM_JustCEDist(Model_Axis_LSTM):
    Name = 'Model_Axis_LSTM_JustCEDist'
    Description = '''
    LSTM Neural Network for SDP Reconstruction
    Only the CEDist is learned
    '''

    def __init__(self,in_main_channels = (4,5,), N_LSTM_nodes = 32,N_LSTM_layers=5 ,N_dense_nodes = 128, **kwargs):
        super(Model_Axis_LSTM_JustCEDist, self).__init__(in_main_channels=in_main_channels, N_LSTM_nodes=N_LSTM_nodes, N_LSTM_layers=N_LSTM_layers, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])
