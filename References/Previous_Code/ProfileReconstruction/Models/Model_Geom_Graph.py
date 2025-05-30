# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import AvgPool1d, MaxPool1d
# from time import time
from   torch_geometric.nn import GATConv #,GCNConv, TAGConv
from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
from   torch_geometric.nn.pool import max_pool_x, avg_pool_x
from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['Chi0','Rp','T0'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys = ['Chi0','Rp','T0'],BatchSize = 256):
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


    
def New_Lout(Lin,Kernel,Stride,Padding,dilation):
    '''
    Takes Lin,Kernel,Stride,Padding,dilation
    Returns the output size of the Conv1d layer
    '''
    return int((Lin + 2*Padding - dilation*(Kernel-1) - 1)/Stride + 1)

            
# Define the model
class Model_Geom_Graph_C(nn.Module):
    Name = 'Model_Geom_Graph_C'
    Description = '''
    Graph Fed Model, Utilises 1dConv to make TraceAnalysis
    '''
    def __init__(self, in_node_channels=5, in_edge_channels = 2, N_Graph_Heads = 16, N_kernels = 16, kernel_size = 10,N_dense_nodes=128, dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        # Input is 20x22 grid
        super(Model_Geom_Graph_C,self).__init__()
        
        # Graph Convolution Layers
        self.Graph1 = GATConv(in_node_channels              , in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        self.Graph2 = GATConv(in_node_channels*N_Graph_Heads, in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        self.Graph3 = GATConv(in_node_channels*N_Graph_Heads, in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        
        self.GraphS = GATConv(in_node_channels*N_Graph_Heads, in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        
        
        # Conv1d Layers ( there would be some math that makes the input to this a padded 500xN_Graph_Heads tensor)
        L_out = 500 # Should be 500 unless something changes
        self.Conv1 = nn.Conv1d(2*in_node_channels*N_Graph_Heads,N_kernels, kernel_size=kernel_size, stride=2, padding=1)
        L_out = New_Lout(L_out,kernel_size,2,1,1)
        self.Conv2 = nn.Conv1d(N_kernels,N_kernels                       , kernel_size=kernel_size, stride=2, padding=1)
        L_out = New_Lout(L_out,kernel_size,2,1,1)
        self.Conv3 = nn.Conv1d(N_kernels,N_kernels                       , kernel_size=kernel_size, stride=2, padding=1)
        L_out = New_Lout(L_out,kernel_size,2,1,1)

        self.ConvActivation = nn.LeakyReLU()
        
        self.AvgPool = AvgPool1d(kernel_size = 10)
        self.MaxPool = MaxPool1d(kernel_size = 10)


        Lout = New_Lout(Lout,10,10,0,1)
        Lout = Lout*2*N_kernels # 2 is for 2 pool types
        

        # Dense Layers
        self.Dense1 = nn.Linear(L_out ,N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # Output Layers
        self.Chi01 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Chi02 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Chi03 = nn.Linear(N_dense_nodes,1)

        self.Rp1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Rp2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Rp3 = nn.Linear(N_dense_nodes,1)
        
        self.T01 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.T02 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.T03 = nn.Linear(N_dense_nodes,1)

        # Activation
        self.DenseActivation = nn.LeakyReLU()
        self.AngleActivation = nn.Tanh()

    def forward(self, Graphs, AuxData,debug = False):
        device = self.Dense1.weight.device
        
        # UnloadGraph
        AllNodes = []
        AllEdges = []
        AllEdgeV = []
        # EvSize   = [] # EvSize is no longer required
        Batching = []
        TotalNNodes = 0
        

        for BatchI,(Nodes,Edges,EdgeV) in enumerate(Graphs):
            AllNodes.append(Nodes)
            AllEdges.append(Edges+TotalNNodes)
            AllEdgeV.append(EdgeV)
            
            # EvSize.append(len(torch.unique(Nodes[:,2]))) # number of unique time steps in the event
            Batching.append(torch.ones(Nodes.shape[0])*BatchI)
            TotalNNodes += Nodes.shape[0]


        AllNodes = torch.cat(AllNodes,dim=0).to(device)
        AllEdges = torch.cat(AllEdges,dim=0).to(device).T
        AllEdgeV = torch.cat(AllEdgeV,dim=0).to(device)

        # Node Info
        # EvSize   = torch.tensor(EvSize)     .to(device).requires_grad_(False)
        Batching = torch.cat(Batching,dim=0).to(device).requires_grad_(False).to(torch.long)
        Timing   = AllNodes[:,2]                       .requires_grad_(False).to(torch.long)

        # Processing

        # Graph Convolution
        out = self.Graph1(AllNodes,AllEdges,edge_attr=AllEdgeV)
        out = self.Graph2(out     ,AllEdges,edge_attr=AllEdgeV)
        out = self.Graph3(out     ,AllEdges,edge_attr=AllEdgeV)
        
        if debug : print(out[Batching == 0])
        
        # Do the Summary by masking edges with NBL >1
        MaskedEdges = AllEdges.T[(AllEdgeV[:,0] > 1)].T
        MaskedEdgeV = AllEdgeV[(AllEdgeV[:,0] > 1)]
        out = self.GraphS(out,MaskedEdges,edge_attr=MaskedEdgeV)
        Max_out,_ = max_pool_x(cluster = Timing, x = out, batch = Batching, size = 500)
        
        Avg_out,_ = avg_pool_x(cluster = Timing, x = out, batch = Batching, size = 500)
        # Reshape for Conv1d
        out = torch.cat([Max_out.view(-1,out.shape[1],500),Avg_out.view(-1,out.shape[1],500)],dim=1)
        if debug: print(out[0,0,...])
        
        # Conv1d
        
        out = self.ConvActivation(self.Conv1(out))
        out = self.ConvActivation(self.Conv2(out))
        out = self.ConvActivation(self.Conv3(out))
        
        Max_out = self.MaxPool(out)
        Avg_out = self.AvgPool(out)
        out = torch.cat([Max_out,Avg_out],dim=1)
        
        # Dense Layers
        out = out.view(out.shape[0],-1)
        
        out = self.DenseActivation(self.Dense1(out))
        out = self.DenseActivation(self.Dense2(out))
        out = self.DenseActivation(self.Dense3(out))

        # print(out)

        # Output Layers
        Chi0 = self.DenseActivation(self.Chi01(out ))
        Chi0 = self.DenseActivation(self.Chi02(Chi0))
        Chi0 = self.AngleActivation(self.Chi03(Chi0))
        
        Rp = self.DenseActivation(self.Rp1(out))
        Rp = self.DenseActivation(self.Rp2(Rp ))
        Rp =                      self.Rp3(Rp )

        T0 = self.DenseActivation(self.T01(out))
        T0 = self.DenseActivation(self.T02(T0 ))
        T0 =                      self.T03(T0 )


        return torch.cat([Chi0,Rp,T0],dim=1)

        
# Define the model
class Model_Geom_Graph_C_S(nn.Module):
    Name = 'Model_Geom_Graph_C_S'
    Description = '''
    Graph Fed Model,C - Utilises 1dConv to make TraceAnalysis, S - Simplified for debuging
    '''
    def __init__(self, in_node_channels=5, in_edge_channels = 2, N_Graph_Heads = 16, N_kernels = 16, kernel_size = 10,N_dense_nodes=128, dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        # Input is 20x22 grid
        super(Model_Geom_Graph_C_S,self).__init__()
        
        # Graph Convolution Layers
        self.Graph1 = GATConv(in_node_channels              , in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        self.GraphS = GATConv(in_node_channels*N_Graph_Heads, in_node_channels, heads = N_Graph_Heads, concat = True, edge_dim = in_edge_channels, fill_value = 0)
        
        
        # Conv1d Layers ( there would be some math that makes the input to this a padded 500xN_Graph_Heads tensor)
        Lout = 500 # Should be 500 unless something changes
        self.Conv1 = nn.Conv1d(in_node_channels*N_Graph_Heads,N_kernels  , kernel_size=kernel_size, stride=2, padding=1)
        Lout = New_Lout(Lout,kernel_size,2,1,1)
        self.Conv2 = nn.Conv1d(N_kernels,N_kernels                       , kernel_size=kernel_size, stride=2, padding=1)
        Lout = New_Lout(Lout,kernel_size,2,1,1)
        self.Conv3 = nn.Conv1d(N_kernels,N_kernels                       , kernel_size=kernel_size, stride=2, padding=1)
        Lout = New_Lout(Lout,kernel_size,2,1,1)

        self.ConvActivation = nn.LeakyReLU()
        
        self.MaxPool = MaxPool1d(kernel_size = 10)
        Lout = New_Lout(Lout,10,10,0,1)
        Lout = Lout*1*N_kernels # 1 is just 1 pool type
        
        # Output Layers
        self.Chi01 = nn.Linear(Lout,N_dense_nodes)
        self.Chi02 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Chi03 = nn.Linear(N_dense_nodes,1)

        self.Rp1 = nn.Linear(Lout,N_dense_nodes)
        self.Rp2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Rp3 = nn.Linear(N_dense_nodes,1)
        
        self.T01 = nn.Linear(Lout,N_dense_nodes)
        self.T02 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.T03 = nn.Linear(N_dense_nodes,1)

        # Activation
        self.DenseActivation = nn.LeakyReLU()
        self.AngleActivation = nn.Tanh()

    def forward(self, Graphs, AuxData,debug = False):
        device = self.Chi01.weight.device
        
        # UnloadGraph
        AllNodes = []
        AllEdges = []
        AllEdgeV = []
        # EvSize   = [] # EvSize is no longer required
        Batching = []
        TotalNNodes = 0
        

        for BatchI,(Nodes,Edges,EdgeV) in enumerate(Graphs):
            AllNodes.append(Nodes)
            AllEdges.append(Edges+TotalNNodes)
            AllEdgeV.append(EdgeV)
            
            # EvSize.append(len(torch.unique(Nodes[:,2]))) # number of unique time steps in the event
            Batching.append(torch.ones(Nodes.shape[0])*BatchI)
            TotalNNodes += Nodes.shape[0]


        AllNodes = torch.cat(AllNodes,dim=0).to(device)
        AllEdges = torch.cat(AllEdges,dim=0).to(device).T
        AllEdgeV = torch.cat(AllEdgeV,dim=0).to(device)

        # Node Info
        # EvSize   = torch.tensor(EvSize)     .to(device).requires_grad_(False)
        Batching = torch.cat(Batching,dim=0).to(device).requires_grad_(False).to(torch.long)
        Timing   = AllNodes[:,2]                       .requires_grad_(False).to(torch.long)

        # Processing

        # Graph Convolution
        out = self.Graph1(AllNodes,AllEdges,edge_attr=AllEdgeV)
        
        if debug : print(out[Batching == 0])
        
        # Do the Summary by masking edges with NBL >1
        MaskedEdges = AllEdges.T[(AllEdgeV[:,0] > 1)].T
        MaskedEdgeV = AllEdgeV[(AllEdgeV[:,0] > 1)]
        out = self.GraphS(out,MaskedEdges,edge_attr=MaskedEdgeV)
        Max_out,_ = max_pool_x(cluster = Timing, x = out, batch = Batching, size = 500)
        
        # Reshape for Conv1d
        out = Max_out.view(-1,out.shape[1],500)
        if debug: print(out[0,0,...])
        
        # Conv1d
        
        out = self.ConvActivation(self.Conv1(out))
        out = self.ConvActivation(self.Conv2(out))
        out = self.ConvActivation(self.Conv3(out))
        
        out = self.MaxPool(out)
        
        # Dense Layers
        out = out.view(out.shape[0],-1)
        if debug: print(out[0,...])
        # Output Layers
        Chi0 = self.DenseActivation(self.Chi01(out ))
        Chi0 = self.DenseActivation(self.Chi02(Chi0))
        Chi0 = self.AngleActivation(self.Chi03(Chi0))
        
        Rp = self.DenseActivation(self.Rp1(out))
        Rp = self.DenseActivation(self.Rp2(Rp ))
        Rp =                      self.Rp3(Rp )

        T0 = self.DenseActivation(self.T01(out))
        T0 = self.DenseActivation(self.T02(T0 ))
        T0 =                      self.T03(T0 )


        return torch.cat([Chi0,Rp,T0],dim=1)

        



        

            
    