# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F

# from time import time
from   torch_geometric.nn import GCNConv, TAGConv,GATv2Conv, EdgeConv
from   torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from   torch_geometric.utils import add_self_loops


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
    

def metric(model,Dataset,device,keys=['x','y','z','SDPPhi','CEDist'],BatchSize = 256):
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
    
class EdgeConvMLP(nn.Module):
    def __init__(self,in_node_channels,out_node_channels,dropout=0.0):
        super(EdgeConvMLP, self).__init__()
        hidden_channels = 2*out_node_channels

        self.L1 = nn.Linear(2*in_node_channels,hidden_channels)
        self.L2 = nn.Linear(   hidden_channels,out_node_channels)
        self.activation_function = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation_function(self.L1(x))
        x = self.dropout(x)
        x = self.activation_function(self.L2(x))
        return x

class Model_Axis_Graph(nn.Module):
    Name = 'Model_Axis_Graph'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution as main mechanicsm.
    Defines Edge conv as first layer to introduce edge information into the network, Edge Information is still used in the GATConv just cause why not?
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        # How many features does edge conv construct for each node: 
        EdgeConv_Features = N_kernels - in_node_channels
        super(Model_Axis_Graph, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.ReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        # Graph Convolution Layers
        self.convE = EdgeConv      (EdgeConvMLP(in_node_channels,EdgeConv_Features), aggr='max')
        self.conv0 = GATv2Conv     (N_kernels        , N_kernels,   heads=N_heads, edge_dim=in_edge_channels, dropout=dropout, fill_value=0.0)
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
        self.XYZ1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.XYZ2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.XYZ3 = nn.Linear(N_dense_nodes//2,3)

        self.SDPPhi1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPPhi3 = nn.Linear(N_dense_nodes//2,1)

        self.CEDist1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.CEDist2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.CEDist3 = nn.Linear(N_dense_nodes//2,1)
        
        self.OutWeights = torch.tensor([1,1,1,1,1])
    

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
        X = self.Conv_Activation(self.convE(AllNodes,AllEdges))
        X = torch.cat([X,AllNodes],dim=1)
        X = self.Conv_Activation(self.conv0(X       ,AllEdges,AllEdgeV))
        # Skip Blocks
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
        XYZ = self.Dense_Activation(self.XYZ1(X))
        XYZ = self.Dense_Activation(self.XYZ2(XYZ))
        XYZ = self.XYZ3(XYZ)

        SDPPhi = self.Dense_Activation(self.SDPPhi1(X))
        SDPPhi = self.Dense_Activation(self.SDPPhi2(SDPPhi))
        SDPPhi = self.Angle_Activation(self.SDPPhi3(SDPPhi))

        CEDist = self.Dense_Activation(self.CEDist1(X))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)
        
        Output = torch.cat([XYZ,SDPPhi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    




class Model_Axis_Graph_JustXYZ(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_JustXYZ'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the XYZ is learned
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_JustXYZ, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1,1,1,0,0])


class Model_Axis_Graph_JustSDPPhi(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_JustSDPPhi'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the SDPPhi is learned
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_JustSDPPhi, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,1,0])


class Model_Axis_Graph_JustCEDist(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_JustCEDist'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_JustCEDist, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])


class Model_Axis_Graph_JustCEDist_TestingSignal(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_JustCEDist_TestingSignal'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Testing wether total charge is the main way to learn CEDist
    Remove charge, and test performance
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_JustCEDist_TestingSignal, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])

    # Instead of using unpacking main and repacking it to call super().forward, just redefine the forward function
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
        # Remove the charge information
        AllNodes[:,4]*=0
        AllNodes[:,5]*=0

        AllEdges = torch.cat(AllEdges,dim=0).to(device).T.to(torch.long)
        AllEdgeV = torch.cat(AllEdgeV,dim=0).to(device)

        Batching = torch.cat(Batching,dim=0).to(device).requires_grad_(False).to(torch.long)

        # Process the Events
        X = self.Conv_Activation(self.convE(AllNodes,AllEdges))
        X = torch.cat([X,AllNodes],dim=1)
        X = self.Conv_Activation(self.conv0(X       ,AllEdges,AllEdgeV))
        # Skip Blocks
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
        XYZ = self.Dense_Activation(self.XYZ1(X))
        XYZ = self.Dense_Activation(self.XYZ2(XYZ))
        XYZ = self.XYZ3(XYZ)

        SDPPhi = self.Dense_Activation(self.SDPPhi1(X))
        SDPPhi = self.Dense_Activation(self.SDPPhi2(SDPPhi))
        SDPPhi = self.Angle_Activation(self.SDPPhi3(SDPPhi))

        CEDist = self.Dense_Activation(self.CEDist1(X))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)
        
        Output = torch.cat([XYZ,SDPPhi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    

class Model_Axis_Graph_JustCEDist_TestingSignal_OnlyPulseDur(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_JustCEDist_TestingSignal_OnlyPulseDur'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Testing wether total charge is the main way to learn CEDist
    Remove charge but keep the pulse duration, and test performance
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_JustCEDist_TestingSignal_OnlyPulseDur, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])

    # Instead of using unpacking main and repacking it to call super().forward, just redefine the forward function
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
        # Remove the charge information
        # AllNodes[:,4]*=0
        AllNodes[:,5]*=0

        AllEdges = torch.cat(AllEdges,dim=0).to(device).T.to(torch.long)
        AllEdgeV = torch.cat(AllEdgeV,dim=0).to(device)

        Batching = torch.cat(Batching,dim=0).to(device).requires_grad_(False).to(torch.long)

        # Process the Events
        X = self.Conv_Activation(self.convE(AllNodes,AllEdges))
        X = torch.cat([X,AllNodes],dim=1)
        X = self.Conv_Activation(self.conv0(X       ,AllEdges,AllEdgeV))
        # Skip Blocks
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
        XYZ = self.Dense_Activation(self.XYZ1(X))
        XYZ = self.Dense_Activation(self.XYZ2(XYZ))
        XYZ = self.XYZ3(XYZ)

        SDPPhi = self.Dense_Activation(self.SDPPhi1(X))
        SDPPhi = self.Dense_Activation(self.SDPPhi2(SDPPhi))
        SDPPhi = self.Angle_Activation(self.SDPPhi3(SDPPhi))

        CEDist = self.Dense_Activation(self.CEDist1(X))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)
        
        Output = torch.cat([XYZ,SDPPhi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    

class Model_Axis_Graph_AngularVelocity(Model_Axis_Graph):
    Name = 'Model_Axis_Graph_AngularVelocity'
    Description = '''
    Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Model requires the Angular Velocity of the Shower Spot as 3rd edge value
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        # How many features does edge conv construct for each node: 
        EdgeConv_Features = N_kernels - in_node_channels
        super(Model_Axis_Graph, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.ReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        # Graph Convolution Layers
        self.convE = EdgeConv      (EdgeConvMLP(in_node_channels,EdgeConv_Features), aggr='max')
        self.conv0 = GATv2Conv     (N_kernels        , N_kernels,   heads=N_heads, edge_dim=in_edge_channels, dropout=dropout, fill_value=0.0)
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
        self.XYZ1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.XYZ2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.XYZ3 = nn.Linear(N_dense_nodes//2,3)

        self.SDPPhi1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPPhi3 = nn.Linear(N_dense_nodes//2,1)

        self.CEDist1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.CEDist2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.CEDist3 = nn.Linear(N_dense_nodes//2,1)
        
        self.OutWeights = torch.tensor([1,1,1,1,1])
    

    # Instead of using unpacking main and repacking it to call super().forward, just redefine the forward function
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
        X = self.Conv_Activation(self.convE(AllNodes,AllEdges))
        X = torch.cat([X,AllNodes],dim=1)
        X = self.Conv_Activation(self.conv0(X       ,AllEdges,AllEdgeV))
        # Skip Blocks
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
        XYZ = self.Dense_Activation(self.XYZ1(X))
        XYZ = self.Dense_Activation(self.XYZ2(XYZ))
        XYZ = self.XYZ3(XYZ)

        SDPPhi = self.Dense_Activation(self.SDPPhi1(X))
        SDPPhi = self.Dense_Activation(self.SDPPhi2(SDPPhi))
        SDPPhi = self.Angle_Activation(self.SDPPhi3(SDPPhi))

        CEDist = self.Dense_Activation(self.CEDist1(X))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)
        
        Output = torch.cat([XYZ,SDPPhi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    



class Model_Axis_Graph_AngularVelocity_JustCEDist(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_JustCEDist'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Model Calculates the Angular Velocity of the Shower Spot and assigns the value to each edge
    Only the CEDist is learned
    '''

    def __init__(self,in_node_channels = 6, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_AngularVelocity_JustCEDist, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])

class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Model Calculates the Angular Velocity of the Shower Spot and assigns the value to each edge
    Using the Angular Velocity as a node feature as well
    '''

    def __init__(self,in_node_channels = 7, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,0,1])





class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist_AngVelOnly(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist_AngVelOnly'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the CEDist is learned
    Model Calculates the Angular Velocity of the Shower Spot and assigns the value to each edge
    Using the Angular Velocity as a node feature as well
    This model will remove information not related to the time-fit to test how well the angular velocity alone performs
    '''

    def __init__(self,in_node_channels = 7, in_edge_channels = 2, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        # Reduction of input channels happens here. 
        # The dataset isnt recalculated, so the inputs are dropped in forward
        
        # 4 in_node_channels are dropped
        in_node_channels -= 4

        
        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustCEDist_AngVelOnly, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        
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

        # Remove the unwanted Nodes
        AllNodes = AllNodes[:,[2,3,6]]

        # Process the Events
        X = self.Conv_Activation(self.convE(AllNodes,AllEdges))
        X = torch.cat([X,AllNodes],dim=1)
        X = self.Conv_Activation(self.conv0(X       ,AllEdges,AllEdgeV))
        # Skip Blocks
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
        XYZ = self.Dense_Activation(self.XYZ1(X))
        XYZ = self.Dense_Activation(self.XYZ2(XYZ))
        XYZ = self.XYZ3(XYZ)

        SDPPhi = self.Dense_Activation(self.SDPPhi1(X))
        SDPPhi = self.Dense_Activation(self.SDPPhi2(SDPPhi))
        SDPPhi = self.Angle_Activation(self.SDPPhi3(SDPPhi))

        CEDist = self.Dense_Activation(self.CEDist1(X))
        CEDist = self.Dense_Activation(self.CEDist2(CEDist))
        CEDist = self.CEDist3(CEDist)
        
        Output = torch.cat([XYZ,SDPPhi,CEDist],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    



# Finish off the variables for the best model found: 

class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustXYZ(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustXYZ'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the XYZ is learned
    Using angular velocity as feature for nodes and edges
    '''    


    def __init__(self,in_node_channels = 7, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):

        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustXYZ, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1,1,1,0,0])

class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustSDPPhi(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustSDPPhi'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the SDPPhi is learned
    Using angular velocity as feature for nodes and edges
    '''    


    def __init__(self,in_node_channels = 7, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):

        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustSDPPhi, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,0,1,0])


# Do dthe XYZ individually
        
class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustX(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustX'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the X coordinate is learned
    Using angular velocity as feature for nodes and edges
    '''    

    def __init__(self,in_node_channels = 7, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustX, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1,0,0,0,0])


class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustY(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustY'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the Y coordinate is learned
    Using angular velocity as feature for nodes and edges
    '''    

    def __init__(self,in_node_channels = 7, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustY, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,1,0,0,0])


class Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustZ(Model_Axis_Graph_AngularVelocity):
    Name = 'Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustZ'
    Description = '''Graph Neural Network for Axis Reconstruction
    Uses GATConv for the Graph Convolution
    Only the Z coordinate is learned
    Using angular velocity as feature for nodes and edges
    '''    

    def __init__(self,in_node_channels = 7, in_edge_channels = 3, N_kernels = 32, N_heads = 4, N_dense_nodes = 128, **kwargs):
        super(Model_Axis_Graph_AngularVelocity_AsNodeAndEdge_JustZ, self).__init__(in_node_channels = in_node_channels, in_edge_channels = in_edge_channels, N_kernels = N_kernels, N_heads = N_heads, N_dense_nodes = N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0,0,1,0,0])
