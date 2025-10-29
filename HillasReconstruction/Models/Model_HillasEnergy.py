# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple # needed for the expected/default values of the functions

# from time import time
# from   torch_geometric.nn import GCNConv, TAGConv,GATv2Conv
# from   torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
# from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['Xmax','LogE','Cherenkov'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        if True and (key == 'LogE'):
            pred  = Pred[:,i] / 0.475 + 16.15
            truth = Truth[:,i] / 0.475 + 16.15
            
            losses[key] = torch.mean((pred - truth)**2 / (truth**2))
        elif False and (key == 'LogE'):
            pred  = Pred[:,i] / 0.475 + 16.15
            truth = Truth[:,i] / 0.475 + 16.15
            
            losses[key] = torch.mean((pred - truth)**2 / (truth))

        elif key == 'LogE':
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
        elif key in ['Xmax','Cherenkov','CherenkovFraction']:
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
        else:
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
            print(f'Warning: Loss for key {key} not defined, using MSE Loss')
    
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
    

def metric(model,Dataset,device,keys=['Xmax','LogE','Cherenkov'],BatchSize = 256):
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

    metrics = {}
    Units = Dataset.Truth_Units
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


class Model_HillasEnergy(nn.Module):
    Name = 'Model_HillasEnergy'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs Xmax and Energy
    '''

    def __init__(self,in_main_channels=(14,) , N_dense_nodes=128, **kwargs):
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        super(Model_HillasEnergy, self).__init__()

        self.Dense_Activation = nn.ReLU()
        self.Dense_Dropout    = nn.Dropout(dropout)

        self.Dense1 = nn.Linear(in_main_channels, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense4 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense5 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)
        
        self.Cherenkov1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Cherenkov2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Cherenkov3 = nn.Linear(N_dense_nodes//2,1)

        
        self.OutWeights = torch.tensor([1,1,1])


    def forward(self,Main,Aux = None):
        device = self.Dense1.weight.device
        Main = Main[0].to(device)
        Main[torch.isnan(Main)] = -1

        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense4(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense5(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Cherenkov = self.Dense_Activation(self.Cherenkov1(Main))
        Cherenkov = self.Dense_Activation(self.Cherenkov2(Cherenkov))
        Cherenkov = self.Cherenkov3(Cherenkov)


        Output = torch.cat([Xmax,Energy,Cherenkov],dim=1)* self.OutWeights.to(device)
        return Output
    
class Model_HillasEnergy_JustXmax(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustXmax'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Xmax
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustXmax, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0, 0])

class Model_HillasEnergy_JustEnergy(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustEnergy'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Energy
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustEnergy, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1, 0])


class Model_HillasEnergy_JustCherenkov(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustCherenkov'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Cherenkov
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustCherenkov, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 0, 1])


class Model_HillasEnergy_JustEnergy_LogLoss(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustEnergy_LogLoss'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Energy
    using MSE Loss
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustEnergy_LogLoss, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1, 0])

class Model_HillasEnergy_JustEnergy_EnergyLoss(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustEnergy_EnergyLoss'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Energy
    using MSE Loss
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustEnergy_EnergyLoss, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1, 0])

class Model_HillasEnergy_JustEnergy_EnergySquaredLoss(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustEnergy_EnergySquaredLoss'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Energy
    using MSE Loss
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustEnergy_EnergySquaredLoss, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1, 0])


class Model_HillasEnergy_JustEnergy_EnergySquaredLoss_SaturationCut(Model_HillasEnergy):
    Name = 'Model_HillasEnergy_JustEnergy_EnergySquaredLoss_SaturationCut'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs only Energy
    using MSE Loss
    '''

    def __init__(self, in_main_channels=(14,), N_dense_nodes=128, **kwargs):
        super(Model_HillasEnergy_JustEnergy_EnergySquaredLoss_SaturationCut, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1, 0])


###########################################################################################
        
class Model_HillasEnergy_Shallow(nn.Module):
    Name = 'Model_HillasEnergyShallow'
    Description = '''
    A simple dense model that takes in the Hillas Parameters and reconstructs Xmax and Energy
    This time its not a deep network, 
    '''

    def __init__(self,in_main_channels=(14,) , N_dense_nodes=128, **kwargs):
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        super(Model_HillasEnergy_Shallow, self).__init__()

        self.Dense_Activation = nn.ReLU()
        self.Dense_Dropout    = nn.Dropout(dropout)

        self.Dense1 = nn.Linear(in_main_channels, N_dense_nodes)
        
        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)
        
        self.Cherenkov1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Cherenkov2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Cherenkov3 = nn.Linear(N_dense_nodes//2,1)

        
        self.OutWeights = torch.tensor([1,1,1])


    def forward(self,Main,Aux = None):
        device = self.Dense1.weight.device
        Main = Main[0].to(device)
        Main[torch.isnan(Main)] = -1

        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        
        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Cherenkov = self.Dense_Activation(self.Cherenkov1(Main))
        Cherenkov = self.Dense_Activation(self.Cherenkov2(Cherenkov))
        Cherenkov = self.Cherenkov3(Cherenkov)


        Output = torch.cat([Xmax,Energy,Cherenkov],dim=1)* self.OutWeights.to(device)
        return Output
