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
    
def Loss(Pred,Truth,keys=['SDPTheta','SDPPhi'],ReturnTensor = True):

    '''
    Takes Truth,Pred in form -> [Theta,Phi] 
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
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
    

def metric(model,Dataset,device,keys = ['SDPTheta','SDPPhi'],BatchSize = 256):
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



# Define the model
class Model_SDP_LSTM_0(nn.Module):
    Name = 'Model_SDP_LSTM_0'
    Description = '''
    Simple LSTM Model for SDP Reconstruction
    '''
    def __init__(self, in_main_channels=1, in_aux_channels_aux=1, N_LSTM_nodes = 32, N_LSTM_layers = 3, N_dense_nodes=128, dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        # Input is 20x22 grid
        super(Model_SDP_LSTM_0,self).__init__()
        # in_Channels should be a tuple of the number of channels for each Main, Only have 1 main here
        in_channels = in_main_channels[0]

        # Define the LSTM
        self.LSTM_input  = nn.LSTM(input_size=in_channels,hidden_size=N_LSTM_nodes,num_layers=N_LSTM_layers,batch_first=True,bidirectional=True)
        self.LSTM_output = nn.LSTM(input_size=N_LSTM_nodes*2,hidden_size=N_LSTM_nodes,num_layers=1,batch_first=True,bidirectional=False) 

        # Aux Analysis
        # Not in this model
        
        # Dense Layers
        self.Dense1 = nn.Linear(N_LSTM_nodes ,N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # Output Layers
        self.Theta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Theta2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Theta3 = nn.Linear(N_dense_nodes,1)

        self.Phi1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Phi2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Phi3 = nn.Linear(N_dense_nodes,1)

        # Activation
        self.DenseActivation = nn.LeakyReLU()
        self.AngleActivation = nn.Tanh()

    def forward(self, Mains, AuxData):
        device = self.LSTM_input.weight_ih_l0.device
        assert len(Mains) == 1, 'Only 1 Main is allowed in this model'
        Mains = Mains[0]
        Mains = Mains.to(device)
        # LSTM
        out = self.LSTM_input(Mains)[0]
        out = self.LSTM_output(out)[0]
        out = out[:,-1,:]

        # Dense Layers
        out = self.DenseActivation(self.Dense1(out))
        out = self.DenseActivation(self.Dense2(out))
        out = self.DenseActivation(self.Dense3(out))

        # Theta
        Theta = self.DenseActivation(self.Theta1(out))
        Theta = self.DenseActivation(self.Theta2(Theta))
        Theta = self.AngleActivation(self.Theta3(Theta))

        # Phi
        Phi = self.DenseActivation(self.Phi1(out))
        Phi = self.DenseActivation(self.Phi2(Phi))
        Phi = self.AngleActivation(self.Phi3(Phi))

        return torch.cat([Theta,Phi],dim=1)
    


        
