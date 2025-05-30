# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F

# from time import time
from   torch_geometric.nn import GCNConv, TAGConv,GATv2Conv
from   torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['Chi0','Rp','T0'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys=['Chi0','Rp','T0'],BatchSize = 256):
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



class GAT_skip_block(nn.Module):
    def __init__(self,in_node_channels,N_kernels,N_heads,N_edge_channels,activation_function,dropout=0.0):
        super(GAT_skip_block, self).__init__()
        
        self.conv1 = GATv2Conv(in_node_channels   , N_kernels, heads=N_heads, edge_dim=N_edge_channels, dropout=dropout, fill_value=0.0)
        self.conv2 = GATv2Conv(N_kernels*N_heads  , N_kernels, heads=N_heads, edge_dim=N_edge_channels, dropout=dropout, fill_value=0.0)
        self.activation_function = activation_function

    def forward(self, x, edge_index, edge_weight):
        x_residual = x
        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation_function(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.activation_function(x)
        x = x + x_residual
        return x
    

class Model_Geom_Graph(nn.Module):
    Name = 'Model_Geom_Graph'
    Description = '''
    Graph Neural Network for SDP Reconstruction
    Uses GATConv for the Graph Convolution
    '''

    def __init__(self,in_node_channels = 4, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        
        super(Model_Geom_Graph, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.ReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        # Graph Convolution Layers
        self.conv0 = GATv2Conv     (in_node_channels, N_kernels, heads=N_heads, edge_dim=in_edge_channels, dropout=dropout, fill_value=0.0)
        self.conv1 = GAT_skip_block(N_kernels*N_heads, N_kernels, N_heads=N_heads, activation_function = self.Conv_Activation , N_edge_channels=in_edge_channels, dropout=dropout)
        self.conv2 = GAT_skip_block(N_kernels*N_heads, N_kernels, N_heads=N_heads, activation_function = self.Conv_Activation , N_edge_channels=in_edge_channels, dropout=dropout)
        self.conv3 = GAT_skip_block(N_kernels*N_heads, N_kernels, N_heads=N_heads, activation_function = self.Conv_Activation , N_edge_channels=in_edge_channels, dropout=dropout)
        self.conv4 = GAT_skip_block(N_kernels*N_heads, N_kernels, N_heads=N_heads, activation_function = self.Conv_Activation , N_edge_channels=in_edge_channels, dropout=dropout)
        
        self.Global_Mean_Pool = global_mean_pool
        self.Global_Max_Pool  = global_max_pool

        # Dense Layers
        self.Dense1 = nn.Linear(2* N_kernels*N_heads, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes       , N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes       , N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)
        self.Rp_1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)
        self.T0_1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.T0_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.T0_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1,1])
    

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        
        # UnloadGraph
        AllNodes = []
        AllEdges = []
        AllEdgeV = []
        Batching = []
        TotalNNodes = 0
        

        for BatchI,(Nodes,Edges,EdgeV) in enumerate(Main):
            AllNodes.append(Nodes)
            AllEdges.append(Edges+TotalNNodes)
            AllEdgeV.append(EdgeV)
            
            Batching.append(torch.ones(Nodes.shape[0])*BatchI)
            TotalNNodes += Nodes.shape[0]


        AllNodes = torch.cat(AllNodes,dim=0).to(device)
        AllEdges = torch.cat(AllEdges,dim=0).to(device).T.to(torch.long)
        AllEdgeV = torch.cat(AllEdgeV,dim=0).to(device)

        Batching = torch.cat(Batching,dim=0).to(device).requires_grad_(False).to(torch.long)

        
        # Process the Events

        X = self.conv0(AllNodes,AllEdges,AllEdgeV)
        X = self.Conv_Activation(X)
        X = self.conv1(X       ,AllEdges,AllEdgeV)
        X = self.conv2(X       ,AllEdges,AllEdgeV)
        X = self.conv3(X       ,AllEdges,AllEdgeV)
        X = self.conv4(X       ,AllEdges,AllEdgeV)

        # Global Pooling
        X = torch.cat([self.Global_Mean_Pool(X,Batching),self.Global_Max_Pool(X,Batching)],dim=1)


        # Dense Layers

        X = self.Dense_Activation(self.Dense1(X))
        X = self.Dense_Activation(self.Dense2(X))
        X = self.Dense_Activation(self.Dense3(X))

        # Output Layers
        Chi0 = self.Dense_Activation(self.Chi0_1(X))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))
        
        Rp   = self.Dense_Activation(self.Rp_1(X))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        T0   = self.Dense_Activation(self.T0_1(X))
        T0   = self.Dense_Activation(self.T0_2(T0))
        T0   = self.T0_3(T0)

        Output = torch.cat([Chi0,Rp,T0],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    




class Model_Geom_Graph_JustChi0(Model_Geom_Graph):
    Name = 'Model_Geom_Graph_JustChi0'
    Description = '''
    Graph Neural Network for SDP Reconstruction
    Uses GATConv for the Graph Convolution
    Only the Theta is learned
    '''

    def __init__(self,in_node_channels = 4, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Geom_Graph_JustChi0, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1,0,0])


class Model_Geom_Graph_JustRp(Model_Geom_Graph):
    Name = 'Model_Geom_Graph_JustRp'
    Description = '''
    Graph Neural Network for SDP Reconstruction
    Uses GATConv for the Graph Convolution
    Only the Rp is learned
    '''

    def __init__(self,in_node_channels = 4, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Geom_Graph_JustRp, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,1,0])


class Model_Geom_Graph_JustT0(Model_Geom_Graph):
    Name = 'Model_Geom_Graph_JustT0'
    Description = '''
    Graph Neural Network for SDP Reconstruction
    Uses GATConv for the Graph Convolution
    Only the T0 is learned
    '''

    def __init__(self,in_node_channels = 4, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Geom_Graph_JustT0, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,1])
