# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
from   torch_geometric.nn import GCNConv, TAGConv
from   torch_geometric.nn.pool import global_mean_pool
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

    def __init__(self,in_channels = 3,NDenseNodes = 16,GCNNodes = 16,Dtype = torch.float32):
        
        super(Model_1_0, self).__init__()
        # Info
        self.Name = 'Model_1_0'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta ->  Predict their Sin and Cos
        Simplest GCN model possible  basically
        '''
        self.Conv1 = GCNConv(in_channels,GCNNodes,dtype=Dtype)
        self.Conv2 = GCNConv(GCNNodes,GCNNodes,dtype=Dtype)
        self.MeanPool = global_mean_pool
        # Dense Layers
        self.PhiDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        


    def forward(self, X, Edge_index, Edge_weight,Batching):
        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Average Pool
        out = self.MeanPool(out,Batching)
        
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)
        


        return torch.cat((Phi,Theta),dim=1)

class Model_1_1(nn.Module):

    def __init__(self,in_channels = 3,NDenseNodes = 16,GCNNodes = 16,Khop = 3,Dtype = torch.float32):
        
        super(Model_1_1, self).__init__()
        # Info
        self.Name = 'Model_1_1'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta ->  Predict their Sin and Cos
        Using TAGConv, supposedly to check for topology
        Also added Self Loops, might help with the K-Hop Conv
        '''
        self.Conv1 = TAGConv(in_channels,GCNNodes,K=Khop,dtype=Dtype)
        self.Conv2 = TAGConv(GCNNodes,GCNNodes,K=Khop,dtype=Dtype)
        self.MeanPool = global_mean_pool
        # Dense Layers
        self.PhiDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        


    def forward(self, X, Edge_index, Edge_weight,Batching):
        if Edge_index.dtype != torch.int64: Edge_index = Edge_index.long()
        # Add self Loops
        Edge_index, Edge_weight = add_self_loops(Edge_index,Edge_weight,num_nodes=X.shape[0])
        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Average Pool
        out = self.MeanPool(out,Batching)
        
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)

        return torch.cat((Phi,Theta),dim=1)
        

class Model_1_2(nn.Module):

    def __init__(self,in_channels = 3,NDenseNodes = 16,GCNNodes = 16,Khop = 3,Dtype = torch.float32):
        
        super(Model_1_2, self).__init__()
        # Info
        self.Name = 'Model_1_2'
        self.Description = '''
        Try to predict the SDP, 
        two parameters: Phi and Theta ->  Predict their Sin and Cos
        Using TAGConv, supposedly to check for topology
        Also added Self Loops, might help with the K-Hop Conv
        Also Will now cut the Edges based on time
        '''
        self.Conv1 = TAGConv(in_channels,GCNNodes,K=Khop,dtype=Dtype)
        self.Conv2 = TAGConv(GCNNodes,GCNNodes,K=Khop,dtype=Dtype)
        self.MeanPool = global_mean_pool
        # Dense Layers
        self.PhiDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.PhiDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ThetaDense1 = nn.Linear(GCNNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.ThetaDense3 = nn.Linear(NDenseNodes,2,dtype=Dtype)

        self.ConvActivation   = nn.LeakyReLU()
        self.DenseActivation  = nn.LeakyReLU()
        


    def forward(self, X, Edge_index, Edge_weight,Batching):
        if Edge_index.dtype != torch.int64: Edge_index = Edge_index.long()

        # Add self Loops
        Edge_index, Edge_weight = add_self_loops(Edge_index,Edge_weight,num_nodes=X.shape[0])
        # Cut Edges
        Mask = Edge_index[1]>Edge_index[0]
        Edge_index = Edge_index[:,Mask]
        Edge_weight = Edge_weight[Mask]
        # Scan
        out = self.ConvActivation(self.Conv1(X,Edge_index,Edge_weight))
        out = self.ConvActivation(self.Conv2(out,Edge_index,Edge_weight))
        # Average Pool
        out = self.MeanPool(out,Batching)
        
        # Predict
        Phi = self.DenseActivation(self.PhiDense1(out))
        Phi = self.DenseActivation(self.PhiDense2(Phi))
        Phi = self.PhiDense3(Phi)
        Theta = self.DenseActivation(self.ThetaDense1(out))
        Theta = self.DenseActivation(self.ThetaDense2(Theta))
        Theta = self.ThetaDense3(Theta)

        return torch.cat((Phi,Theta),dim=1)
        

# Define the Loss Function
    
def Loss(Pred,Truth):
    '''
    Takes Truth,Pred in form -> [CosPhi, SinPhi, CosTheta, SinTheta]
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    # # Phi
    CosPhiTruth = Truth[0]
    SinPhiTruth = Truth[1]
    CosPhiPred = Pred[0]
    SinPhiPred = Pred[1]
    CosPhiLoss = F.mse_loss(CosPhiPred,CosPhiTruth)
    SinPhiLoss = F.mse_loss(SinPhiPred,SinPhiTruth)

    # Theta
    CosThetaTruth = Truth[2]
    SinThetaTruth = Truth[3]
    CosThetaPred = Pred[2]
    SinThetaPred = Pred[3]
    CosThetaLoss = F.mse_loss(CosThetaPred,CosThetaTruth)
    SinThetaLoss = F.mse_loss(SinThetaPred,SinThetaTruth)

    # Sum up
    Total_Loss = CosPhiLoss + SinPhiLoss + CosThetaLoss + SinThetaLoss
    Phi_Loss   = CosPhiLoss + SinPhiLoss
    Theta_Loss = CosThetaLoss + SinThetaLoss
    return Total_Loss,Phi_Loss,Theta_Loss



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






