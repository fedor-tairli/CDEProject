# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F

# from time import time
# from   torch_geometric.nn import GCNConv, TAGConv,GATv2Conv
# from   torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
# from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['X','Y','Z'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys=['X','Y','Z'],BatchSize = 256):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the 68% containment range of the angular deviation
    '''
    assert keys == ['X','Y','Z'], 'Only X,Y,Z keys are supported for this metric'

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
    # Preds  = Dataset.Unnormalise_Truth(Preds )
    # Truths = Dataset.Unnormalise_Truth(Truths)

    # unique way to construct a metric - produce angular deviation between truth and prediction
    # and return the 68% containment range
    dot_product = torch.sum(Truths * Preds, dim=-1)
    norm_a = torch.norm(Truths, dim=-1)
    norm_b = torch.norm(Preds, dim=-1)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    angle = torch.acos(cos_theta)

    Units = Dataset.Truth_Units
    # Combined metric for the 'XYZ' keys 
    metrics = {}
    metrics['X'] = torch.quantile(torch.abs(angle[:]),0.68)
    metrics['Y'] = metrics['X']
    metrics['Z'] = metrics['X']
    
    
    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return metrics



class Conv_Skip_Block(nn.Module):
    def __init__(self, in_channels, N_kernels,activation_function, kernel_size=3, padding=1, stride=1, dropout=0.0):
        assert in_channels == N_kernels, 'Input and Output Channels should be same'
        super(Conv_Skip_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, N_kernels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(N_kernels, N_kernels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation_function = activation_function
    def forward(self, x):
        x_residual = x
        x = self.activation_function(self.conv1(x))
        x = self.activation_function(self.conv2(x))
        return x + x_residual




class Model_SDP_3vector(nn.Module):
    Name = 'Model_SDP_3vector'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for one telescope
    This model does not do max pooling
    '''

    def __init__(self,in_main_channels = (2,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_SDP_3vector, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout2d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0_large = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        self.conv_0_small = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=(1,0)) # Out=> (N, N_kernels, 20, 22)
        self.conv_0       = nn.Conv2d(N_kernels*2     , N_kernels, kernel_size=3, padding=1    ) # Out=> (N, N_kernels, 20, 22)
        self.BN_0         = nn.BatchNorm2d(N_kernels)
        self.conv_1 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 20, 20)
        self.BN_1   = nn.BatchNorm2d(N_kernels)
        # Pool here to (N, N_kernels, 10, 10)
        self.conv_2 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 10, 10)
        self.BN_2   = nn.BatchNorm2d(N_kernels)
        # Pooling here to (N, N_kernels, 5, 5)
        self.conv_3 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_3   = nn.BatchNorm2d(N_kernels)

        self.conv_4 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_4   = nn.BatchNorm2d(N_kernels)

        self.conv_5 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_5   = nn.BatchNorm2d(N_kernels)


        # Reshape to (N, N_kernels*5*5)
        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Vector1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Vecotr2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.Vector3   = nn.Linear(N_dense_nodes//2,3)

        self.OutWeights = torch.tensor([1,1])

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 1, 'Only one Main is expected'
        Main = Main[0].to(device)
        

        Main_L = self.Conv_Activation(self.conv_0_large(Main))
        Main_S = self.Conv_Activation(self.conv_0_small(Main))
        
        Main = torch.cat([Main_L,Main_S],dim=1)
        Main = self.Conv_Dropout(self.Conv_Activation(self.conv_0(Main)))
        Main = self.BN_0(Main)
        Main = self.Conv_Dropout(self.conv_1(Main))
        Main = self.BN_1(Main)
        
        Main = self.Conv_Dropout(self.conv_2(Main))
        Main = self.BN_2(Main)
        
        Main = self.conv_3(Main)
        Main = self.BN_3(Main)

        Main = self.Conv_Dropout(self.conv_4(Main))
        Main = self.BN_4(Main)

        Main = self.Conv_Dropout(self.conv_5(Main))
        Main = self.BN_5(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        Vector = self.Dense_Activation(self.Vector1(Main))
        Vector = self.Dense_Activation(self.Vecotr2(Vector))
        Vector = self.Angle_Activation(self.Vector3(Vector))
        # Output = Output*self.OutWeights.to(device)

        return Vector    
    




