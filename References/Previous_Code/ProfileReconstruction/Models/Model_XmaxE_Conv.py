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
    
# def Loss(Pred,Truth,keys=['Xmax','LogE'],ReturnTensor = True):

#     '''
#     Calculates MSE Loss for all the keys in the keys list
#     '''
#     assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
#     Truth = Truth.to(Pred.device)
#     # Calculate Loss
#     losses = {}
#     for i,key in enumerate(keys):
#         losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
    
#     losses['Total'] = sum(losses.values())
#     if ReturnTensor: return losses
#     else:
#         losses = {key:loss.item() for key,loss in losses.items()}
#         return losses

# Do loss that is scaled by LogE
def Loss(Pred,Truth,keys=['Xmax','LogE'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    device = Pred.device
    Ebins   = torch.tensor([-1e99,-9.99849319e-01, -9.59852517e-01, -9.19855714e-01, -8.79858911e-01,
        -8.39862108e-01, -7.99865365e-01, -7.59868562e-01, -7.19871759e-01,
        -6.79874957e-01, -6.39878154e-01, -5.99881351e-01, -5.59884548e-01,
        -5.19887745e-01, -4.79890972e-01, -4.39894170e-01, -3.99897397e-01,
        -3.59900594e-01, -3.19903791e-01, -2.79906988e-01, -2.39910200e-01,
        -1.99913412e-01, -1.59916610e-01, -1.19919814e-01, -7.99230188e-02,
        -3.99262235e-02,  7.05718994e-05,  4.00673673e-02,  8.00641626e-02,
         1.20060958e-01,  1.60057753e-01,  2.00054556e-01,  2.40051344e-01,
         2.80048132e-01,  3.20044935e-01,  3.60041738e-01,  4.00038540e-01,
         4.40035313e-01,  4.80032116e-01,  5.20028889e-01,  5.60025692e-01,
         6.00022495e-01,  6.40019298e-01,  6.80016100e-01,  7.20012903e-01,
         7.60009706e-01,  8.00006509e-01,  8.40003252e-01,  8.80000055e-01,
         9.19996858e-01,  9.59993660e-01,  9.99990463e-01,1e99]).to(device)
    Escales = torch.tensor([1e99,4440., 4785., 5292., 6074., 5557., 7032., 7742., 8135., 7728.,
        8276., 9183., 9798., 6272., 3257., 3185., 3804., 3572., 3714.,
        3711., 4101., 3996., 4020., 3950., 4253., 4295., 1936., 2150.,
        1955., 1918., 2090., 1845., 1902., 2080., 2015., 2010., 2127.,
        1936., 1840., 1938., 1981., 1967., 1912., 1989., 2033., 2102.,
        1977., 1912., 1952., 1963., 1880.,1e99]).to(device)
    
    # Energy is in Truth [:,1]
    Truth = Truth.to(Pred.device)
    bin_indices = torch.bucketize(Truth[:,1],Ebins)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = ((Pred[:,i]-Truth[:,i])**2/Escales[bin_indices]).mean()
    
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
    

def metric(model,Dataset,device,keys = ['Xmax','LogE'],BatchSize = 256):
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

class Model_XmaxE_Conv_3d_Distances(nn.Module):
    Name = 'Model_XmaxE_Conv_3d_Distances'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    Will have Distances and Heights for every pixel along
    Will not be predicting the Axis SDPPhi and Distance to Core
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_XmaxE_Conv_3d_Distances,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(3        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
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
        
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TracesMain     = torch.zeros(NEvents,100 ,20,22)
        DistancesMain  = torch.zeros(NEvents,100 ,20,22)
        HeightsMain    = torch.zeros(NEvents,100 ,20,22)
        StartMain      = torch.zeros(NEvents,1   ,20,22)

        Main_Traces    = torch.zeros(NEvents,1100,20,22)
        Main_Distances = torch.zeros(NEvents,1100,20,22)
        Main_Heights   = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces    = torch.cat(list(map(lambda x : x[0], Graph)))
        Distances = torch.cat(list(map(lambda x : x[1], Graph)))
        Heights   = torch.cat(list(map(lambda x : x[2], Graph)))
        Xs        = torch.cat(list(map(lambda x : x[3], Graph)))
        Ys        = torch.cat(list(map(lambda x : x[4], Graph)))
        Pstart    = torch.cat(list(map(lambda x : x[5], Graph)))

        TracesMain    [EventIndices,:,Xs,Ys] = Traces
        StartMain     [EventIndices,0,Xs,Ys] = Pstart
        DistancesMain [EventIndices,:,Xs,Ys] = Distances.repeat_interleave(100).reshape(Traces.shape)
        HeightsMain   [EventIndices,:,Xs,Ys] = Heights  .repeat_interleave(100).reshape(Traces.shape)

        DistancesMain = DistancesMain*(TracesMain>0).float()
        HeightsMain   = HeightsMain  *(TracesMain>0).float()

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main_Traces   .scatter_(1,indices,TracesMain)
        Main_Distances.scatter_(1,indices,DistancesMain)
        Main_Heights  .scatter_(1,indices,HeightsMain)

        Main_Traces    = Main_Traces   .unfold(1,10,10)
        Main_Distances = Main_Distances.unfold(1,10,10)
        Main_Heights   = Main_Heights  .unfold(1,10,10)

        Main_Traces    = Main_Traces   .sum(-1)
        Main_Distances = Main_Distances.max(-1)
        Main_Heights   = Main_Heights  .max(-1)

        Main_Traces    = Main_Traces          [:,:80,:,:].unsqueeze(1).to(device)
        Main_Distances = Main_Distances.values[:,:80,:,:].unsqueeze(1).to(device)
        Main_Heights   = Main_Heights.values  [:,:80,:,:].unsqueeze(1).to(device)

        Main_Traces   [torch.isnan(Main_Traces)]    = -1
        Main_Distances[torch.isnan(Main_Distances)] = -1
        Main_Heights  [torch.isnan(Main_Heights)]   = -1

        Main = torch.cat((Main_Traces,Main_Distances,Main_Heights),dim=1)
        

        # Actual Forward        
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

        
        
        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)

        
        return torch.cat([Xmax,E],dim=1)


class Model_XmaxE_Conv_3d_Distances_JustXmax(nn.Module):
    Name = 'Model_XmaxE_Conv_3d_Distances_JustXmax'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    Will have Distances and Heights for every pixel along
    Will not be predicting the Axis SDPPhi and Distance to Core
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_XmaxE_Conv_3d_Distances_JustXmax,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(3        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
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
        
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TracesMain     = torch.zeros(NEvents,100 ,20,22)
        DistancesMain  = torch.zeros(NEvents,100 ,20,22)
        HeightsMain    = torch.zeros(NEvents,100 ,20,22)
        StartMain      = torch.zeros(NEvents,1   ,20,22)

        Main_Traces    = torch.zeros(NEvents,1100,20,22)
        Main_Distances = torch.zeros(NEvents,1100,20,22)
        Main_Heights   = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces    = torch.cat(list(map(lambda x : x[0], Graph)))
        Distances = torch.cat(list(map(lambda x : x[1], Graph)))
        Heights   = torch.cat(list(map(lambda x : x[2], Graph)))
        Xs        = torch.cat(list(map(lambda x : x[3], Graph)))
        Ys        = torch.cat(list(map(lambda x : x[4], Graph)))
        Pstart    = torch.cat(list(map(lambda x : x[5], Graph)))

        TracesMain    [EventIndices,:,Xs,Ys] = Traces
        StartMain     [EventIndices,0,Xs,Ys] = Pstart
        DistancesMain [EventIndices,:,Xs,Ys] = Distances.repeat_interleave(100).reshape(Traces.shape)
        HeightsMain   [EventIndices,:,Xs,Ys] = Heights  .repeat_interleave(100).reshape(Traces.shape)

        DistancesMain = DistancesMain*(TracesMain>0).float()
        HeightsMain   = HeightsMain  *(TracesMain>0).float()

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main_Traces   .scatter_(1,indices,TracesMain)
        Main_Distances.scatter_(1,indices,DistancesMain)
        Main_Heights  .scatter_(1,indices,HeightsMain)

        Main_Traces    = Main_Traces   .unfold(1,10,10)
        Main_Distances = Main_Distances.unfold(1,10,10)
        Main_Heights   = Main_Heights  .unfold(1,10,10)

        Main_Traces    = Main_Traces   .sum(-1)
        Main_Distances = Main_Distances.max(-1)
        Main_Heights   = Main_Heights  .max(-1)

        Main_Traces    = Main_Traces          [:,:80,:,:].unsqueeze(1).to(device)
        Main_Distances = Main_Distances.values[:,:80,:,:].unsqueeze(1).to(device)
        Main_Heights   = Main_Heights.values  [:,:80,:,:].unsqueeze(1).to(device)

        Main_Traces   [torch.isnan(Main_Traces)]    = -1
        Main_Distances[torch.isnan(Main_Distances)] = -1
        Main_Heights  [torch.isnan(Main_Heights)]   = -1

        Main = torch.cat((Main_Traces,Main_Distances,Main_Heights),dim=1)
        

        # Actual Forward        
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

        
        
        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        E = torch.zeros(Xmax.shape).to(Xmax.device)
        
        return torch.cat([Xmax,E],dim=1)


class Model_XmaxE_Conv_3d_Distances_JustLogE(nn.Module):
    Name = 'Model_XmaxE_Conv_3d_Distances_JustLogE'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    Will have Distances and Heights for every pixel along
    Will not be predicting the Axis SDPPhi and Distance to Core
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_XmaxE_Conv_3d_Distances_JustLogE,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(3        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 80x20x20
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
        
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TracesMain     = torch.zeros(NEvents,100 ,20,22)
        DistancesMain  = torch.zeros(NEvents,100 ,20,22)
        HeightsMain    = torch.zeros(NEvents,100 ,20,22)
        StartMain      = torch.zeros(NEvents,1   ,20,22)

        Main_Traces    = torch.zeros(NEvents,1100,20,22)
        Main_Distances = torch.zeros(NEvents,1100,20,22)
        Main_Heights   = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces    = torch.cat(list(map(lambda x : x[0], Graph)))
        Distances = torch.cat(list(map(lambda x : x[1], Graph)))
        Heights   = torch.cat(list(map(lambda x : x[2], Graph)))
        Xs        = torch.cat(list(map(lambda x : x[3], Graph)))
        Ys        = torch.cat(list(map(lambda x : x[4], Graph)))
        Pstart    = torch.cat(list(map(lambda x : x[5], Graph)))

        TracesMain    [EventIndices,:,Xs,Ys] = Traces
        StartMain     [EventIndices,0,Xs,Ys] = Pstart
        DistancesMain [EventIndices,:,Xs,Ys] = Distances.repeat_interleave(100).reshape(Traces.shape)
        HeightsMain   [EventIndices,:,Xs,Ys] = Heights  .repeat_interleave(100).reshape(Traces.shape)

        DistancesMain = DistancesMain*(TracesMain>0).float()
        HeightsMain   = HeightsMain  *(TracesMain>0).float()

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main_Traces   .scatter_(1,indices,TracesMain)
        Main_Distances.scatter_(1,indices,DistancesMain)
        Main_Heights  .scatter_(1,indices,HeightsMain)

        Main_Traces    = Main_Traces   .unfold(1,10,10)
        Main_Distances = Main_Distances.unfold(1,10,10)
        Main_Heights   = Main_Heights  .unfold(1,10,10)

        Main_Traces    = Main_Traces   .sum(-1)
        Main_Distances = Main_Distances.max(-1)
        Main_Heights   = Main_Heights  .max(-1)

        Main_Traces    = Main_Traces          [:,:80,:,:].unsqueeze(1).to(device)
        Main_Distances = Main_Distances.values[:,:80,:,:].unsqueeze(1).to(device)
        Main_Heights   = Main_Heights.values  [:,:80,:,:].unsqueeze(1).to(device)

        Main_Traces   [torch.isnan(Main_Traces)]    = -1
        Main_Distances[torch.isnan(Main_Distances)] = -1
        Main_Heights  [torch.isnan(Main_Heights)]   = -1

        Main = torch.cat((Main_Traces,Main_Distances,Main_Heights),dim=1)
        

        # Actual Forward        
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

        
        
        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)

        Xmax = torch.zeros(E.shape).to(E.device)

        return torch.cat([Xmax,E],dim=1)


class Model_XmaxE_Conv_3d_Distances_JustXmax_HalfBins(nn.Module):
    Name = 'Model_XmaxE_Conv_3d_Distances_JustXmax_HalfBins'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    Will have Distances and Heights for every pixel along
    Will not be predicting the Axis SDPPhi and Distance to Core
    Same As before, but i reduce the rebinning to 5 time steps instead of 10
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_XmaxE_Conv_3d_Distances_JustXmax_HalfBins,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(3        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 160x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      160x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      160x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 20x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 10x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*20+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

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
        
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TracesMain     = torch.zeros(NEvents,100 ,20,22)
        DistancesMain  = torch.zeros(NEvents,100 ,20,22)
        HeightsMain    = torch.zeros(NEvents,100 ,20,22)
        StartMain      = torch.zeros(NEvents,1   ,20,22)

        Main_Traces    = torch.zeros(NEvents,1100,20,22)
        Main_Distances = torch.zeros(NEvents,1100,20,22)
        Main_Heights   = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces    = torch.cat(list(map(lambda x : x[0], Graph)))
        Distances = torch.cat(list(map(lambda x : x[1], Graph)))
        Heights   = torch.cat(list(map(lambda x : x[2], Graph)))
        Xs        = torch.cat(list(map(lambda x : x[3], Graph)))
        Ys        = torch.cat(list(map(lambda x : x[4], Graph)))
        Pstart    = torch.cat(list(map(lambda x : x[5], Graph)))

        TracesMain    [EventIndices,:,Xs,Ys] = Traces
        StartMain     [EventIndices,0,Xs,Ys] = Pstart
        DistancesMain [EventIndices,:,Xs,Ys] = Distances.repeat_interleave(100).reshape(Traces.shape)
        HeightsMain   [EventIndices,:,Xs,Ys] = Heights  .repeat_interleave(100).reshape(Traces.shape)

        DistancesMain = DistancesMain*(TracesMain>0).float()
        HeightsMain   = HeightsMain  *(TracesMain>0).float()

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main_Traces   .scatter_(1,indices,TracesMain)
        Main_Distances.scatter_(1,indices,DistancesMain)
        Main_Heights  .scatter_(1,indices,HeightsMain)

        Main_Traces    = Main_Traces   .unfold(1,5,5)
        Main_Distances = Main_Distances.unfold(1,5,5)
        Main_Heights   = Main_Heights  .unfold(1,5,5)

        Main_Traces    = Main_Traces   .sum(-1)
        Main_Distances = Main_Distances.max(-1)
        Main_Heights   = Main_Heights  .max(-1)

        Main_Traces    = Main_Traces          [:,:160,:,:].unsqueeze(1).to(device)
        Main_Distances = Main_Distances.values[:,:160,:,:].unsqueeze(1).to(device)
        Main_Heights   = Main_Heights.values  [:,:160,:,:].unsqueeze(1).to(device)

        Main_Traces   [torch.isnan(Main_Traces)]    = -1
        Main_Distances[torch.isnan(Main_Distances)] = -1
        Main_Heights  [torch.isnan(Main_Heights)]   = -1

        Main = torch.cat((Main_Traces,Main_Distances,Main_Heights),dim=1)
        

        # Actual Forward        
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

        
        
        Xmax = self.FC_Activation(  self.Xmax1(X)  )
        Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        Xmax = self.Xmax3(Xmax)

        # E    = self.FC_Activation(  self.E1(X)  )
        # E    = self.FC_Activation(  self.E2(E)  )
        # E    = self.E3(E)
        E = torch.zeros(Xmax.shape).to(Xmax.device)
        
        return torch.cat([Xmax,E],dim=1)


class Model_XmaxE_Conv_3d_Distances_JustLogE_HalfBins(nn.Module):
    Name = 'Model_XmaxE_Conv_3d_Distances_JustLogE_HalfBins'
    Description = '''
    Simple Conv3D Model for Geometry Reconstruction from 3D Grid
    Will have Distances and Heights for every pixel along
    Will not be predicting the Axis SDPPhi and Distance to Core
    '''

    def __init__(self, in_aux_channels_aux=3, out_channels=7, N_kernels = 32, N_dense_nodes=128, init_type = None,DropOut_rate = 0,dtype = torch.float32,**kwargs):
        # Pretty much kwargs are not used here, but kept for compatibility
        self.kwargs = kwargs
        super(Model_XmaxE_Conv_3d_Distances_JustLogE_HalfBins,self).__init__()
        
        # Expects a set of graph data which will be transformed to 3D grid 
        # Main Shape will be (N,80,20,22) # Dont forget to unsquish to account for extra dimension of the channel
        # No Auxilliary Data for now

        # 3D Convolutional Layers
        self.Conv1 = nn.Conv3d(3        ,N_kernels,kernel_size=3,stride=1,padding=(1,1,0)) # Padding to reduce to 160x20x20
        self.Conv2 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      160x20x20
        self.Conv3 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1      ) # Padding to keep      160x20x20
        self.Pool1 = nn.MaxPool3d(kernel_size=(8,2,2),stride=(8,2,2)) # Reduces to 20x10x10

        
        self.Conv4 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Conv5 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Conv6 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 20x10x10
        self.Pool2 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 10x5x5

        self.Conv7 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Conv8 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Conv9 = nn.Conv3d(N_kernels,N_kernels,kernel_size=3,stride=1,padding=1) # Keeps 10x5x5
        self.Pool3 = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)) # Reduces to 5x2x2


        # Auxilliary Data Processing
        self.Aux1 = nn.Linear(N_kernels*20+in_aux_channels_aux,N_dense_nodes)
        self.Aux2 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Aux3 = nn.Linear(N_dense_nodes,N_dense_nodes)

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
        
        # Unpack The Graph Data to Main
        device = self.Conv1.weight.device
        Aux = Aux.to(device) # Dont need to do anything else to it
        NEvents = len(Graph)
        TracesMain     = torch.zeros(NEvents,100 ,20,22)
        DistancesMain  = torch.zeros(NEvents,100 ,20,22)
        HeightsMain    = torch.zeros(NEvents,100 ,20,22)
        StartMain      = torch.zeros(NEvents,1   ,20,22)

        Main_Traces    = torch.zeros(NEvents,1100,20,22)
        Main_Distances = torch.zeros(NEvents,1100,20,22)
        Main_Heights   = torch.zeros(NEvents,1100,20,22)
        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces    = torch.cat(list(map(lambda x : x[0], Graph)))
        Distances = torch.cat(list(map(lambda x : x[1], Graph)))
        Heights   = torch.cat(list(map(lambda x : x[2], Graph)))
        Xs        = torch.cat(list(map(lambda x : x[3], Graph)))
        Ys        = torch.cat(list(map(lambda x : x[4], Graph)))
        Pstart    = torch.cat(list(map(lambda x : x[5], Graph)))

        TracesMain    [EventIndices,:,Xs,Ys] = Traces
        StartMain     [EventIndices,0,Xs,Ys] = Pstart
        DistancesMain [EventIndices,:,Xs,Ys] = Distances.repeat_interleave(100).reshape(Traces.shape)
        HeightsMain   [EventIndices,:,Xs,Ys] = Heights  .repeat_interleave(100).reshape(Traces.shape)

        DistancesMain = DistancesMain*(TracesMain>0).float()
        HeightsMain   = HeightsMain  *(TracesMain>0).float()

        indices = (torch.arange(100).reshape(1,-1,1,1)+StartMain).long()
        Main_Traces   .scatter_(1,indices,TracesMain)
        Main_Distances.scatter_(1,indices,DistancesMain)
        Main_Heights  .scatter_(1,indices,HeightsMain)

        Main_Traces    = Main_Traces   .unfold(1,5,5)
        Main_Distances = Main_Distances.unfold(1,5,5)
        Main_Heights   = Main_Heights  .unfold(1,5,5)

        Main_Traces    = Main_Traces   .sum(-1)
        Main_Distances = Main_Distances.max(-1)
        Main_Heights   = Main_Heights  .max(-1)

        Main_Traces    = Main_Traces          [:,:160,:,:].unsqueeze(1).to(device)
        Main_Distances = Main_Distances.values[:,:160,:,:].unsqueeze(1).to(device)
        Main_Heights   = Main_Heights.values  [:,:160,:,:].unsqueeze(1).to(device)

        Main_Traces   [torch.isnan(Main_Traces)]    = -1
        Main_Distances[torch.isnan(Main_Distances)] = -1
        Main_Heights  [torch.isnan(Main_Heights)]   = -1

        Main = torch.cat((Main_Traces,Main_Distances,Main_Heights),dim=1)
        

        # Actual Forward        
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

        
        
        # Xmax = self.FC_Activation(  self.Xmax1(X)  )
        # Xmax = self.FC_Activation(  self.Xmax2(Xmax)  )
        # Xmax = self.Xmax3(Xmax)

        E    = self.FC_Activation(  self.E1(X)  )
        E    = self.FC_Activation(  self.E2(E)  )
        E    = self.E3(E)

        Xmax = torch.zeros(E.shape).to(E.device)

        return torch.cat([Xmax,E],dim=1)
