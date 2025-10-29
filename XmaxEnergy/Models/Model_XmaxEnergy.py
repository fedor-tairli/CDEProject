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
    
def Loss(Pred,Truth,keys=['Xmax','LogE'],ReturnTensor = True):

    '''
    Calculates MSE Loss for all the keys in the keys list
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    Truth = Truth.to(Pred.device)
    # Calculate Loss
    losses = {}
    for i,key in enumerate(keys):

        if True and (key == 'LogE'):
            pred  = Pred[:,i] / 0.475 + 16.15
            truth = Truth[:,i] / 0.475 + 16.15
            
            losses[key] = torch.mean((pred - truth)**2 / (truth**2))
        
        elif key == 'LogE':
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
        elif key in ['Xmax','Cherenkov','CherenkovFraction']:
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
        else:
            losses[key] = F.mse_loss(Pred[:,i],Truth[:,i])
            print(f'Warning: Loss for key {key} not defined, using MSE Loss')
    
    
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
    

def metric(model,Dataset,device,keys=['Xmax','LogE'],BatchSize = 256):
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

    metrics = {}
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


class Model_XmaxEnergy_Conv3d(nn.Module):
    Name = 'Model_XmaxEnergy_Conv3d'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_Conv3d, self).__init__()

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

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = torch.tensor([1,1])
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
        # Main = self.Conv_Dropout(Main)
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

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return Output
    
class Model_SDP_Conv3d_JustXmax(Model_XmaxEnergy_Conv3d):
    Name = 'Model_SDP_Conv3d_JustXmax'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(1,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv3d_JustXmax, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])


class Model_SDP_Conv3d_JustEnergy(Model_XmaxEnergy_Conv3d):
    Name = 'Model_SDP_Conv3d_JustEnergy'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(1,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv3d_JustEnergy, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])

class Model_XmaxEnergy_Conv3d_JustXmax(Model_XmaxEnergy_Conv3d):
    Name = 'Model_XmaxEnergy_Conv3d_JustXmax'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(1,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_XmaxEnergy_Conv3d_JustXmax, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])


class Model_XmaxEnergy_Conv3d_JustEnergy(Model_XmaxEnergy_Conv3d):
    Name = 'Model_XmaxEnergy_Conv3d_JustEnergy'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(1,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_XmaxEnergy_Conv3d_JustEnergy, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])



####################################################################################################################




class Model_XmaxEnergy_ManFeatures(nn.Module):
    Name = 'Model_XmaxEnergy_ManFeatures'
    Description = '''
    Takes in the input of the graph traces, but converts them to manual features instead
    Simple 2D convolutions into the dense layers
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_ManFeatures, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout3d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)


        # In is the (N,1,40,20,22)
        # Manufacture Features to make (N,3,20,22)

        self.conv0 = nn.Conv2d(in_channels=in_main_channels*3, out_channels=N_kernels, kernel_size=3, padding = (1,0) , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.conv2 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1)
        self.conv3 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1)

        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = torch.tensor([1,1])

    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TracesCharges   = Traces.sum(dim=1)
        TracesMax       = Traces.max(dim=1).values
        # Traces Centroids is the Signal*TimeBin / Charge
        TracesCentroids = torch.sum(Traces*torch.arange(Traces.shape[1]).reshape(1,-1),dim=1) / (TracesCharges + 1e-6) + Pstart.reshape(-1)
        
        # Now place the features into the Main
        Main      = torch.zeros(NEvents,3 ,20,22)
        
        Main[EventIndices,0,Xs,Ys] = TracesCharges
        Main[EventIndices,1,Xs,Ys] = TracesMax
        Main[EventIndices,2,Xs,Ys] = TracesCentroids
        Main[torch.isnan(Main)] = -1
        Main = Main.to(device)

        # Process the Data
        Main = self.Conv_Activation(self.conv0(Main))
        Main = self.Conv_Activation(self.conv1(Main))
        Main = self.Conv_Activation(self.conv2(Main))
        Main = self.Conv_Activation(self.conv3(Main))
        
        
        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return Output
    


class Model_XmaxEnergy_LSTMFeatures(nn.Module):
    Name = 'Model_XmaxEnergy_LSTMFeatures'
    Description = '''
    Takes in the input of the graph traces, but converts them to manual features instead
    Simple 2D convolutions into the dense layers
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128,N_LSTM_nodes = 5,N_LSTM_Layers = 3, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_LSTMFeatures, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout3d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)

        self.N_LSTM_nodes = N_LSTM_nodes

        # In is the (N,1,40,20,22)
        # Use LSTM to produce Features to make (N,N_LSTM_nodes+1,20,22) +1 for the Pstart
        self.LSTM = nn.LSTM(input_size=40, hidden_size=N_LSTM_nodes, num_layers=N_LSTM_Layers, batch_first=True, bidirectional=True)

        self.conv0 = nn.Conv2d(in_channels=N_LSTM_nodes+1, out_channels=N_kernels, kernel_size=3, padding = (1,0) , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.conv2 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1)
        self.conv3 = nn.Conv2d(in_channels=N_kernels, out_channels=N_kernels, kernel_size=3, padding = 1 , stride = 1)

        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = torch.tensor([1,1])

    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        Traces = Traces.to(device)
        LSTM_out, (h_n, c_n) = self.LSTM(Traces)

        TracesLSTM = LSTM_out[:,0] # Using Bi-dir becuase most of the signal is at the beginining and then can take the first output of the LSTM

        TracesLSTM = TracesLSTM.to('cpu')

        # Now place the features into the Main
        Main      = torch.zeros(NEvents,self.N_LSTM_nodes+1 ,20,22)

        for i in range(self.N_LSTM_nodes):
            Main[EventIndices,i,Xs,Ys] = TracesLSTM[:]
        Main[EventIndices,self.N_LSTM_nodes,Xs,Ys] = Pstart
        
        Main[torch.isnan(Main)] = -1
        Main = Main.to(device)

        # Process the Data
        Main = self.Conv_Activation(self.conv0(Main))
        Main = self.Conv_Activation(self.conv1(Main))
        Main = self.Conv_Activation(self.conv2(Main))
        Main = self.Conv_Activation(self.conv3(Main))
        
        
        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return Output


####################################################################################################################
    

class Model_XmaxEnergy_Conv2d(nn.Module):
    Name = 'Model_XmaxEnergy_Conv2d'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Takes in the standard conv3d data, but flattens it to conv2d
    Uses standard Conv2d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        if in_main_channels != 1: raise ValueError('in_main_channels expected to be (1,) for Conv2d Model')
        in_main_channels = 2
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_Conv2d, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout2d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)


        # In is (N,2,20,22)
        self.Conv0 = nn.Conv2d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,0) , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.Conv1 = nn.Conv2d(in_channels=N_kernels       , out_channels=N_kernels, kernel_size=3, padding = 1     , stride = 1) # Out=> (N, N_kernels, 20, 20)
        self.Conv2 = nn.Conv2d(in_channels=N_kernels       , out_channels=N_kernels, kernel_size=3, padding = 1     , stride = 1)
        self.Conv3 = nn.Conv2d(in_channels=N_kernels       , out_channels=N_kernels, kernel_size=3, padding = 1     , stride = 1)

        
        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = torch.tensor([1,1])


    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        
        Main      = torch.zeros(NEvents,2,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it.

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        
        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        Trace_Centroid = torch.sum(Traces*torch.arange(Traces.shape[1]).reshape(1,-1),dim=1) / (Traces.sum(dim=1) + 1e-6) + Pstart.reshape(-1)
        Trace_Charge   = Traces.sum(dim=1)

        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()
        
        Main[EventIndices,0,Xs,Ys] = Trace_Charge
        Main[EventIndices,1,Xs,Ys] = Trace_Centroid

        Main[torch.isnan(Main)] = -1
        Main = Main.to(device)

        # Process the Data
        Main = self.Conv_Activation(self.Conv0(Main))
        # Main = self.Conv_Dropout(Main)
        Main = self.Conv_Activation(self.Conv1(Main))
        Main = self.Conv_Activation(self.Conv2(Main))
        Main = self.Conv_Activation(self.Conv3(Main))

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return Output
