# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
from   torch_geometric.nn import GCNConv, TAGConv,GATConv
from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import numpy as np
import os

import sys
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# add code path into the path
sys.path.append(Paths.code_path)

from TrainingModule import IterateInBatches




# Define the model

class Model_1_0(nn.Module):

    def __init__(self,in_channels = 4,NDenseNodes = 32,GCNNodes = 16,Dtype = torch.float32):
        
        super(Model_1_0, self).__init__()
        # Info
        self.Name = 'Model_1_0'
        self.Description = '''
        Try to predict Chi0 and Rp using a simple GCN
        '''
        self.Conv1 = GCNConv(in_channels,GCNNodes,dtype=Dtype)
        self.Conv2 = GCNConv(GCNNodes,GCNNodes,dtype=Dtype)
        self.MeanPool = global_mean_pool
        # Dense Layers
        self.Chi0Dense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        


    def forward(self, X, Edge_index, Edge_weight,Batching):
        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Average Pool
        out = self.MeanPool(out,Batching)
        
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)

class Model_1_1(nn.Module):

    def __init__(self,in_channels = 4,NDenseNodes = 32,GCNNodes = 16,Dtype = torch.float32):
        
        super(Model_1_1, self).__init__()
        # Info
        self.Name = 'Model_1_1'
        self.Description = '''
        Add self loops and only check propagation downstream
        Also add max pooling and concatenate
        '''
        self.Conv1 = GCNConv(in_channels,GCNNodes,dtype=Dtype)
        self.Conv2 = GCNConv(GCNNodes,GCNNodes,dtype=Dtype)
        self.MeanPool = global_mean_pool
        self.MaxPool  = global_max_pool

        # Dense Layers
        self.Chi0Dense1 = nn.Linear(GCNNodes*2,NDenseNodes,dtype=Dtype) # *2 for the Pool Concatenation
        self.Chi0Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(GCNNodes*2,NDenseNodes,dtype=Dtype)  # *2 for the Pool Concatenation
        self.RpDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        



    def forward(self, X, Edge_index, Edge_weight,Batching):
        # Add self loops
        Edge_index, Edge_weight = add_self_loops(Edge_index, Edge_weight, num_nodes=X.shape[0])
        # Cut Edges
        Mask = Edge_index[1]>Edge_index[0]
        Edge_index = Edge_index[:,Mask]
        Edge_weight = Edge_weight[Mask]

        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Average Pool
        outMean = self.MeanPool(out,Batching)
        outMax = self.MaxPool(out,Batching)
        out = torch.cat((outMean,outMax),dim=1)

        
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)

class Model_1_2(nn.Module):

    def __init__(self,in_channels = 4,NDenseNodes = 32,GCNNodes = 16,attention_heads = 1,Dtype = torch.float32):
        # assert attention_heads == 1, 'Not sure if this works with more than 1 head'
        super(Model_1_2, self).__init__()
        # Info
        self.Name = 'Model_1_2'
        self.Description = '''
        Add self loops and only check propagation downstream (removed)
        Also add max pooling and concatenate
        And Swwitch to GATConv
        '''
        self.Conv1 = GATConv(in_channels,GCNNodes,heads = attention_heads,add_self_loops=False,dtype=Dtype)
        self.Conv2 = GATConv(GCNNodes,GCNNodes,heads = attention_heads,add_self_loops=False,dtype=Dtype)
        self.MeanPool = global_mean_pool
        self.MaxPool  = global_max_pool

        # Dense Layers
        self.Chi0Dense1 = nn.Linear(GCNNodes*2*attention_heads,NDenseNodes,dtype=Dtype) # *2 for the Pool Concatenation
        self.Chi0Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(GCNNodes*2*attention_heads,NDenseNodes,dtype=Dtype)  # *2 for the Pool Concatenation
        self.RpDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        



    def forward(self, X, Edge_index, Edge_weight,Batching):
        # Add self loops
        Edge_index, Edge_weight = add_self_loops(Edge_index, Edge_weight, num_nodes=X.shape[0])
        # # Cut Edges
        # Mask = Edge_index[1]>Edge_index[0]
        # Edge_index = Edge_index[:,Mask]
        # Edge_weight = Edge_weight[Mask]

        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Pool
        outMean = self.MeanPool(out,Batching)
        outMax = self.MaxPool(out,Batching)
        out = torch.cat((outMean,outMax),dim=1)
        
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)



# Define the Loss Function
    
def Loss(Pred,Truth):
    '''
    Takes Truth,Pred in form -> [CosPhi, SinPhi, CosTheta, SinTheta]
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    # Chi0
    Chi0Truth = Truth[0]
    Chi0Pred = Pred[0]
    Chi0Loss = F.mse_loss(Chi0Pred,Chi0Truth)

    # Rp
    RpTruth = Truth[1]
    RpPred = Pred[1]
    RpLoss = F.mse_loss(RpPred,RpTruth)

    
    # Sum up
    Total_Loss = Chi0Loss + RpLoss
    return Total_Loss,Chi0Loss,RpLoss



# Define the Validation Function

def validate(model,Dataset,Loss,device,normStateOut=None):# NormStateOut is not used
    '''
    Takes model, Dataset, Loss Function, device - Dataset is defined as ProcessingDataset in the Dataset.py
    Iterates over the dataset using the IterateInBatches function
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    batchN = 0 
    with torch.no_grad():
        val_loss = 0
        val_loss_Phi = 0
        val_loss_Theta = 0
        for DatasetEventIndex,BatchGraphBatching,BatchFeatures,BatchEdges,BatchEdgesWeights,BatchTruth in IterateInBatches(Dataset,256):
            if BatchEdges.shape[1] == 0: continue
            BatchGraphBatching = BatchGraphBatching  .to(device)
            BatchFeatures      = BatchFeatures       .to(device)
            BatchEdges         = BatchEdges          .to(device)
            BatchEdgesWeights  = BatchEdgesWeights   .to(device)
            BatchTruth         = BatchTruth          .to(device)
            
            predictions = model(BatchFeatures,BatchEdges,BatchEdgesWeights,BatchGraphBatching)
            loss,PhiLoss,ThetaLoss = Loss(predictions,BatchTruth)
            val_loss += loss.item()
            val_loss_Phi += PhiLoss.item()
            val_loss_Theta += ThetaLoss.item()
            batchN += 1
    val_loss = val_loss/batchN
    val_loss_Phi = val_loss_Phi/batchN
    val_loss_Theta = val_loss_Theta/batchN
    return val_loss,val_loss_Phi,val_loss_Theta






