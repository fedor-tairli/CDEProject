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
    
def Loss(Pred,Truth,keys=['x','y','z','SDPPhi','CEDist','Xmax','LogE'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys = ['x','y','z','SDPPhi','CEDist','Xmax','LogE'],BatchSize = 256):
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



class Model_Axis_Conv_3d(nn.Module):
    Name = 'Model_Axis_Conv_3d'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # Output Layers
        self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        Axis = self.FC_Activation(  self.Axis1(X)  )
        Axis = self.FC_Activation(  self.Axis2(Axis)  )
        Axis = self.Axis3(Axis)

        Phi  = self.FC_Activation(  self.Phi1(X)  )
        Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        Dist = self.FC_Activation(  self.Dist1(X)  )
        Dist = self.FC_Activation(  self.Dist2(Dist)  )
        Dist = self.Dist3(Dist)

        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)



class Model_Axis_Conv_3d_JustXmax(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustXmax'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustXmax,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        Axis = torch.zeros(Xmax.shape[0],3).to(device)
        Phi  = torch.zeros(Xmax.shape[0],1).to(device)
        Dist = torch.zeros(Xmax.shape[0],1).to(device)
        E    = torch.zeros(Xmax.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustLogE(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustLogE'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustLogE,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        Axis = torch.zeros(E.shape[0],3).to(device)
        Phi  = torch.zeros(E.shape[0],1).to(device)
        Dist = torch.zeros(E.shape[0],1).to(device)
        Xmax = torch.zeros(E.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustAxis(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustAxis'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustAxis,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        Axis = self.FC_Activation(  self.Axis1(X)  )
        Axis = self.FC_Activation(  self.Axis2(Axis)  )
        Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E    = torch.zeros(Axis.shape[0],1).to(device)
        Phi  = torch.zeros(Axis.shape[0],1).to(device)
        Dist = torch.zeros(Axis.shape[0],1).to(device)
        Xmax = torch.zeros(Axis.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustSDPPhi(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustSDPPhi'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustSDPPhi,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        Phi  = self.FC_Activation(  self.Phi1(X)  )
        Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E     = torch.zeros(Phi.shape[0],1).to(device)
        Axis  = torch.zeros(Phi.shape[0],3).to(device)
        Dist  = torch.zeros(Phi.shape[0],1).to(device)
        Xmax  = torch.zeros(Phi.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustCEDist(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustCEDist'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustCEDist,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        Dist = self.FC_Activation(  self.Dist1(X)  )
        Dist = self.FC_Activation(  self.Dist2(Dist)  )
        Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E     = torch.zeros(Dist.shape[0],1).to(device)
        Axis  = torch.zeros(Dist.shape[0],3).to(device)
        Phi  = torch.zeros(Dist.shape[0],1).to(device)
        Xmax  = torch.zeros(Dist.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    


# Now follow the models without the station data being input
    

class Model_Axis_Conv_3d_JustXmax_NoStation(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustXmax_NoStation'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustXmax_NoStation,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        # X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        Axis = torch.zeros(Xmax.shape[0],3).to(device)
        Phi  = torch.zeros(Xmax.shape[0],1).to(device)
        Dist = torch.zeros(Xmax.shape[0],1).to(device)
        E    = torch.zeros(Xmax.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustLogE_NoStation(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustLogE_NoStation'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustLogE_NoStation,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        # X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        Axis = torch.zeros(E.shape[0],3).to(device)
        Phi  = torch.zeros(E.shape[0],1).to(device)
        Dist = torch.zeros(E.shape[0],1).to(device)
        Xmax = torch.zeros(E.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustAxis_NoStation(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustAxis_NoStation'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustAxis_NoStation,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        # X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        Axis = self.FC_Activation(  self.Axis1(X)  )
        Axis = self.FC_Activation(  self.Axis2(Axis)  )
        Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E    = torch.zeros(Axis.shape[0],1).to(device)
        Phi  = torch.zeros(Axis.shape[0],1).to(device)
        Dist = torch.zeros(Axis.shape[0],1).to(device)
        Xmax = torch.zeros(Axis.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustSDPPhi_NoStation(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustSDPPhi_NoStation'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustSDPPhi_NoStation,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        # self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        # X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        Phi  = self.FC_Activation(  self.Phi1(X)  )
        Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        # Dist = self.FC_Activation(  self.Dist1(X)  )
        # Dist = self.FC_Activation(  self.Dist2(Dist)  )
        # Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E     = torch.zeros(Phi.shape[0],1).to(device)
        Axis  = torch.zeros(Phi.shape[0],3).to(device)
        Dist  = torch.zeros(Phi.shape[0],1).to(device)
        Xmax  = torch.zeros(Phi.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

class Model_Axis_Conv_3d_JustCEDist_NoStation(nn.Module):
    Name = 'Model_Axis_Conv_3d_JustCEDist_NoStation'
    Description = '''
    Simple Conv3D Model for Reconstruction from 3D Grid
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_Axis_Conv_3d_JustCEDist_NoStation,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(1        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      80x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 10x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 5x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 2x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*8,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

        # # Output Layers
        # self.Axis1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Axis2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Axis3 = nn.Linear(N_dense_nodes//4,3)

        # self.Phi1  = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Phi2  = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Phi3  = nn.Linear(N_dense_nodes//4,1)

        self.Dist1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Dist2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        self.Dist3 = nn.Linear(N_dense_nodes//4,1)

        # self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.Xmax2 = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.Xmax3 = nn.Linear(N_dense_nodes//4,1)

        # self.E1    = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        # self.E2    = nn.Linear(N_dense_nodes//2,N_dense_nodes//4)
        # self.E3    = nn.Linear(N_dense_nodes//4,1)

        
        # Activation Functions
        self.Conv_Activation   = nn.LeakyReLU()
        self.FC_Activation     = nn.LeakyReLU()
        self.Angle_Activation  = nn.Tanh()
        self.NoActivation      = nn.Identity()

    def forward(self,Graph,Aux):
        # time0 = time()
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TraceMain = torch.zeros(NEvents,100 ,20,22)
        StartMain = torch.zeros(NEvents,1   ,20,22)
        Main      = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        # print(N_pixels_in_event.dtype)
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        Main = Main.unfold(1,10,10)
        Main = Main.sum(-1)
        Main = Main[:,:80,:,:].unsqueeze(1).to(device)
        
        Main[torch.isnan(Main)] = -1
        
        
        # Process the Data
        X = self.Conv_Activation(  self.Conv1(Main )  )
        X = self.Conv_Activation(  self.Conv2(  X  )  )
        X = self.Conv_Activation(  self.Conv3(  X  )  )
        X = self.Pool1(X)
        X = self.Conv_Activation(  self.Conv4(  X  )  )
        X = self.Conv_Activation(  self.Conv5(  X  )  )
        X = self.Conv_Activation(  self.Conv6(  X  )  )
        X = self.Pool2(X)
        X = self.Conv_Activation(  self.Conv7(  X  )  )
        X = self.Conv_Activation(  self.Conv8(  X  )  )
        X = self.Conv_Activation(  self.Conv9(  X  )  )
        X = self.Pool3(X)

        X = X.view(X.shape[0],-1)
        # X = torch.cat([X,Aux],dim=1)

        X = self.FC_Activation(  self.Aux1(X)  )
        X = self.FC_Activation(  self.Aux2(X)  )
        X = self.FC_Activation(  self.Aux3(X)  )

        # Axis = self.FC_Activation(  self.Axis1(X)  )
        # Axis = self.FC_Activation(  self.Axis2(Axis)  )
        # Axis = self.Axis3(Axis)

        # Phi  = self.FC_Activation(  self.Phi1(X)  )
        # Phi  = self.FC_Activation(  self.Phi2(Phi)  )
        # Phi  = self.Angle_Activation(  self.Phi3(Phi)  )

        Dist = self.FC_Activation(  self.Dist1(X)  )
        Dist = self.FC_Activation(  self.Dist2(Dist)  )
        Dist = self.Dist3(Dist)

        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        
        # Spoof Some values for loss to not break
        E     = torch.zeros(Dist.shape[0],1).to(device)
        Axis  = torch.zeros(Dist.shape[0],3).to(device)
        Phi   = torch.zeros(Dist.shape[0],1).to(device)
        Xmax  = torch.zeros(Dist.shape[0],1).to(device)

        return torch.cat([Axis,Phi,Dist,Xmax,E],dim=1)
    

    
