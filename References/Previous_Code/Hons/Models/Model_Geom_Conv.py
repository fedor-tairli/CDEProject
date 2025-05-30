# Importing the libraries
import numpy as np
import os
import sys

import torch 
import torch.nn as nn
import torch.nn.functional as F
# from   torch_geometric.nn import GCNConv, TAGConv,GATConv
# from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
# from   torch_geometric.utils import add_self_loops


# Define the Loss Function
    
def Loss(Pred,Truth,keys=['Chi0','Rp','T0'],ReturnTensor = True):

    '''
    Takes Truth,Pred in form -> [Theta,Phi] 
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
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
        for _,BatchMains,BatchAux,BatchTruth,_ in Dataset:
            # Send to device
            BatchMains = [Main.to(device) for Main in BatchMains]
            BatchAux   = BatchAux.to(device)
            
            Preds .append( model(BatchMains,BatchAux).to('cpu') )
            Truths.append(BatchTruth.to('cpu'))

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
        for _,BatchMains,BatchAux,BatchTruth,_ in Dataset:
            # Send to device
            BatchMains = [Main.to(device) for Main in BatchMains]
            BatchAux   = BatchAux.to(device)
            
            Preds .append( model(BatchMains,BatchAux).to('cpu') )
            Truths.append(BatchTruth.to('cpu'))

    Preds  = torch.cat(Preds ,dim=0)
    Truths = torch.cat(Truths,dim=0)
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
class Model_Geom_Conv_0(nn.Module):
    Name = 'Model_Geom_Conv_0'
    Description = '''
    Simple Conv2D Model for SDP Reconstruction
    3x 5x5 and 3x 3x3 Conv2d Layers followed by 3x 3x3 Conv2d Layers into 6x Fully Connected Layers
    '''
    def __init__(self, in_channels=1, in_channels_aux=1, out_channels=1, N_kernels = 16, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32):
        # Input is 20x22 grid
        super(Model_Geom_Conv_0,self).__init__()
        # in_Channels should be a tuple of the number of channels for each Main, Only have 1 main here
        in_channels = in_channels[0]
        # 5x5 Conv2d
        self.Scan1 = nn.Conv2d(in_channels,N_kernels,kernel_size=5,stride=1,padding=(2,1)) # Reduces to 20x20
        
        self.Conv1 = nn.Conv2d(N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))   # Keeps 20x20
        self.Conv2 = nn.Conv2d(N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))   # Keeps 20x20
        self.Conv3 = nn.Conv2d(N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))   # Keeps 20x20
        self.Conv4 = nn.Conv2d(N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))   # Keeps 20x20
        self.Conv5 = nn.Conv2d(N_kernels,N_kernels,kernel_size=5,stride=1,padding=(2,2))   # Keeps 20x20

        # Fully Connected Layers
        self.FC1 = nn.Linear(20*20*N_kernels,N_dense_nodes)
        self.FC2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.FC3 = nn.Linear(N_dense_nodes,N_dense_nodes)

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

        # Activation Functions
        self.Conv_Activation = nn.LeakyReLU()
        self.FC_Activation   = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation    = nn.Identity()


    def forward(self,Mains,Aux):
        # Assertions
        assert len(Mains) == 1, f'Expected 1 Main, got {len(Mains)}'
        assert Mains[0].shape[2:] == (20,22), f'Expected Main Shape (N,1,20,22), got {Mains[0].shape}'

        Main = Mains[0]
        # Convolutional Layers
        X = self.Scan1(Main)
        X = self.Conv_Activation(  self.Conv1(X)  )
        X = self.Conv_Activation(  self.Conv2(X)  )
        X = self.Conv_Activation(  self.Conv3(X)  )
        X = self.Conv_Activation(  self.Conv4(X)  )
        X = self.Conv_Activation(  self.Conv5(X)  )

        # Flatten
        X = X.view(X.shape[0],-1)

        # Fully Connected Layers
        X = self.FC_Activation(  self.FC1(X)  )
        X = self.FC_Activation(  self.FC2(X)  )
        X = self.FC_Activation(  self.FC3(X)  )

        # Output Layers
        Chi0 = self.FC_Activation( self.Chi01(X) )
        Chi0 = self.FC_Activation( self.Chi02(Chi0) )
        Chi0 = self.Angle_Activation( self.Chi03(Chi0) )

        Rp = self.FC_Activation( self.Rp1(X) )
        Rp = self.FC_Activation( self.Rp2(Rp) )
        Rp = self.NoActivation( self.Rp3(Rp) )

        T0 = self.FC_Activation( self.T01(X) )
        T0 = self.FC_Activation( self.T02(T0) )
        T0 = self.NoActivation( self.T03(T0) )

        # print('Chi0',Chi0)
        # print('Rp',Rp)
        # print('T0',T0)

        return torch.cat([Chi0,Rp,T0],dim=1)


