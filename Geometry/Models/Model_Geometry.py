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
def DistBasedWeights(Dist):
    return torch.nn.functional.sigmoid(-0.3*Dist+3)

def Loss(Pred,Truth,keys=('X','Y','Axis_X','Axis_Y','Axis_Z'),ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''

    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    Dist = torch.sqrt(Truth[:,3]**2 + Truth[:,4]**2)
    weights = DistBasedWeights(Dist)


    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.mse_loss(Pred[:,i],Truth[:,i],weight=weights)
    
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
    

def metric(model,Dataset,device,keys=('X','Y','Axis_X','Axis_Y','Axis_Z'),BatchSize = 256):
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
    
    # Check for SDP_double definition, 
    metrics = {}
    if 'X' in keys and 'Y' in keys and 'Axis_X' in keys and 'Axis_Y' in keys and 'Axis_Z' in keys:
        
        # X and Y get the 68% containment range of the absolute difference
        metrics['X']        = torch.quantile(torch.abs(Preds[:,0]-Truths[:,0]),0.68)
        metrics['Y']        = torch.quantile(torch.abs(Preds[:,1]-Truths[:,1]),0.68)
        # Axis should gets the 68% containment range of the angular difference
        # Normalise the vector for the 
        Preds_Axis = Preds[:,2:5]
        Truths_Axis = Truths[:,2:5]

        Preds_Axis  = Preds_Axis / torch.norm(Preds_Axis,dim=1,keepdim=True)
        Truths_Axis = Truths_Axis / torch.norm(Truths_Axis,dim=1,keepdim=True)
        AngleDiv = torch.acos(torch.sum(Preds_Axis * Truths_Axis,dim=1))

        metrics['Axis_X'] = torch.quantile(torch.abs(AngleDiv),0.68) * 180/torch.pi
        metrics['Axis_Y'] = metrics['Axis_X'] # Axis_Y is the same as Axis_X
        metrics['Axis_Z'] = metrics['Axis_X'] # Axis_Z is the same as Axis_X
        
    
    else:
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











class Conv_Skip_Block_3d(nn.Module):
    def __init__(self, in_channels, N_kernels, activation_function, kernel_size:Union[int,Tuple[int,int,int]]=3, padding:Union[int,Tuple[int,int,int]]=1, stride=1, dropout=0.0):
        assert in_channels == N_kernels, 'Input and Output Channels should be the same'
        super(Conv_Skip_Block_3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, N_kernels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv3d(N_kernels, N_kernels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation_function = activation_function
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        x_residual = x
        x = self.activation_function(self.conv1(x))
        x = self.dropout(x)
        x = self.activation_function(self.conv2(x))
        x = self.dropout(x)
        return x + x_residual





class Model_Geometry_Conv3d_CameraPlane_Axis(nn.Module):
    Name = 'Model_Geometry_Conv3d_CameraPlane_Axis'
    Description = '''
    Convolutional Neural Network for the Full Geometry Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution

    The  model is deisgned to produce the geometry normalised to the camera plane, using the axis based parametrisation.
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_Geometry_Conv3d_CameraPlane_Axis, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout3d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)


        self.conv0 = nn.Conv3d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,1,0) , stride = 1) # Out=> (N, N_kernels, 40, 20, 20)
        self.Conv1 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv2 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv3 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)

        self.Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Cords1 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Cords2 = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Cords3 = nn.Linear(N_dense_nodes//2, 2)

        self.Axis1  = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Axis2  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Axis3  = nn.Linear(N_dense_nodes//2,3)

        self.OutWeights = torch.tensor([1,1,1,1,1])


    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        
        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100 ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it.

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        # Main = Main.unfold(1,10,10)
        # Main = Main.sum(-1)
        # Main = Main[:,:80,:,:].unsqueeze(1).to(device)
       
       # Dont need to rebin this, because our dimensions are already small

       # Just take the first 40 bins
        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1

        # Process the Data
        Main = self.Conv_Activation(self.conv0(Main))
        Main = self.Conv_Dropout(Main)
        Main = self.Conv1(Main)
        Main = self.Conv2(Main)
        Main = self.Conv3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        
        # Cords branch
        Cords = self.Dense_Activation(self.Cords1(Main))
        Cords = self.Dense_Dropout(Cords)
        Cords = self.Dense_Activation(self.Cords2(Cords))
        Cords = self.Dense_Dropout(Cords)
        Cords = self.Cords3(Cords)

        # Axis branch
        Axis = self.Dense_Activation(self.Axis1(Main))
        Axis = self.Dense_Dropout(Axis)
        Axis = self.Dense_Activation(self.Axis2(Axis))
        Axis = self.Dense_Dropout(Axis)
        Axis = self.Angle_Activation(self.Axis3(Axis))

        # Concatenate outputs
        Output = torch.cat([Cords, Axis], dim=1)
        
        return Output
