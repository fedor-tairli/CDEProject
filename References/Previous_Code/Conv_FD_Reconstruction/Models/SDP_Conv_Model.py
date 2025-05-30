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
    
def Loss(Pred,Truth,keys=['SDPTheta','SDPPhi'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys=['SDPTheta','SDPPhi'],BatchSize = 256):
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


class Model_SDP_Conv(nn.Module):
    Name = 'Model_SDP_Conv'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers
    Reconstruction is done for triple telescopes
    First split, and run the input conv on 3 telescopes individually
    Then concatenate the results and run  main conv layers
    '''

    def __init__(self,in_main_channels = (3,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        
        super(Model_SDP_Conv, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes

        self.conv_inp_1 = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=1)
        self.conv_inp_2 = nn.Conv2d(N_kernels  , N_kernels, kernel_size=3, padding=1)
        self.conv_inp_3 = nn.Conv2d(N_kernels  , N_kernels, kernel_size=3, padding=1)
        self.BN_inp_1   = nn.BatchNorm2d(N_kernels)

        # Concatenate to the main chunk should be (N, N_kernels, 60, 22)
        # Run pooling here # Should be (N, N_kernels, 30, 11)
        self.pool = nn.MaxPool2d(2,2)

        # Main Convolution Layers
        self.conv_main_1 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.conv_main_2 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.conv_main_3 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.BN_main_1   = nn.BatchNorm2d(N_kernels)
        # One more pool here # Should be (N, N_kernels, 15, 5)
        # More Convolutions
        self.conv_main_4 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.conv_main_5 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.conv_main_6 = nn.Conv2d(N_kernels, N_kernels, kernel_size=3, padding=1)
        self.BN_main_2   = nn.BatchNorm2d(N_kernels)

        # Final Pooling # Should be (N, N_kernels, 7, 2)

        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*7*2, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.SDPTheta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 1, 'Only one Main is expected'
        Main = Main[0].to(device)
        
        # Split Main into 3 telescopes
        Main1 = Main[:,:, 0:20,:]
        Main2 = Main[:,:,20:40,:]
        Main3 = Main[:,:,40:60,:]
        
        # Run Convolution on each telescope
        Main1 = self.Conv_Activation(self.conv_inp_1(Main1))
        Main1 = self.Conv_Activation(self.conv_inp_2(Main1))
        Main1 = self.Conv_Activation(self.conv_inp_3(Main1))
        Main1 = self.BN_inp_1(Main1)
        Main2 = self.Conv_Activation(self.conv_inp_1(Main2))
        Main2 = self.Conv_Activation(self.conv_inp_2(Main2))
        Main2 = self.Conv_Activation(self.conv_inp_3(Main2))
        Main2 = self.BN_inp_1(Main2)
        Main3 = self.Conv_Activation(self.conv_inp_1(Main3))
        Main3 = self.Conv_Activation(self.conv_inp_2(Main3))
        Main3 = self.Conv_Activation(self.conv_inp_3(Main3))
        Main3 = self.BN_inp_1(Main3)

        # Concatenate the results
        Main = torch.cat([Main1,Main2,Main3],dim=2)
        # print(f'Expected shape (N,N_kernels, 60,22) Got -> {Main.shape}')
        
        Main = self.pool(Main)
        # print(f'Expected shape (N,N_kernels, 30,11) Got -> {Main.shape}')

        # Run Main Convolution Layers
        Main = self.Conv_Activation(self.conv_main_1(Main))
        Main = self.Conv_Activation(self.conv_main_2(Main))
        Main = self.Conv_Activation(self.conv_main_3(Main))
        Main = self.BN_main_1(Main)
        Main = self.pool(Main)
        # print(f'Expected shape (N,N_kernels, 15,6) Got -> {Main.shape}')

        Main = self.Conv_Activation(self.conv_main_4(Main))
        Main = self.Conv_Activation(self.conv_main_5(Main))
        Main = self.Conv_Activation(self.conv_main_6(Main))
        Main = self.BN_main_2(Main)
        Main = self.pool(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # print(f'Expected shape (N,{32*7*2}) Got -> {Main.shape}')

        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        Theta = self.Dense_Activation(self.SDPTheta1(Main))
        Theta = self.Dense_Activation(self.SDPTheta2(Theta))
        Theta = self.Angle_Activation(self.SDPTheta3(Theta))

        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        Output = torch.cat([Theta,Phi],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output
    

class Model_SDP_Conv_JustTheta(Model_SDP_Conv):
    Name = 'Model_SDP_Conv_JustTheta'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_JustTheta, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])


class Model_SDP_Conv_JustPhi(Model_SDP_Conv):
    Name = 'Model_SDP_Conv_JustPhi'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_JustPhi, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])


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



class Model_SDP_Conv_Residual(nn.Module):
    Name = 'Model_SDP_Conv_Residual'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for triple telescopes
    First split, and run the input conv on 3 telescopes individually
    Then concatenate the results and run  main conv layers
    '''

    def __init__(self,in_main_channels = (3,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.0
        
        super(Model_SDP_Conv_Residual, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0 = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        
        self.conv_1 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 20, 20)
        # Concat and Pool here to (N, N_kernels, 30, 10)
        self.conv_2 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 15, 10)
        # Pooling here to (N, N_kernels, 7, 5)
        self.conv_3 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        
        self.pool = nn.MaxPool2d(2,2)

        # Reshape to (N, N_kernels*7*5)
        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*15*5, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.SDPTheta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 1, 'Only one Main is expected'
        Main = Main[0].to(device)
        
        # Split Main into 3 telescopes
        Main1 = Main[:,:, 0:20,:]
        Main2 = Main[:,:,20:40,:]
        Main3 = Main[:,:,40:60,:]
        
        # Run Convolution on each telescope
        Main1 = self.Conv_Activation(self.conv_0(Main1))
        Main1 = self.conv_1(Main1)
        Main2 = self.Conv_Activation(self.conv_0(Main2))
        Main2 = self.conv_1(Main2)
        Main3 = self.Conv_Activation(self.conv_0(Main3))
        Main3 = self.conv_1(Main3)
        
        # Concatenate the results
        Main = torch.cat([Main1,Main2,Main3],dim=2)
        Main = self.pool(Main)
        Main = self.conv_2(Main)
        Main = self.pool(Main)
        Main = self.conv_3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        Theta = self.Dense_Activation(self.SDPTheta1(Main))
        Theta = self.Dense_Activation(self.SDPTheta2(Theta))
        Theta = self.Angle_Activation(self.SDPTheta3(Theta))

        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        Output = torch.cat([Theta,Phi],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output    
    
class Model_SDP_Conv_Residual_JustTheta(Model_SDP_Conv_Residual):
    Name = 'Model_SDP_Conv_Residual_JustTheta'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_JustTheta, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])

class Model_SDP_Conv_Residual_JustPhi(Model_SDP_Conv_Residual):
    Name = 'Model_SDP_Conv_Residual_JustPhi'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_JustPhi, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])



class Model_SDP_Conv_Residual_SingleTel(nn.Module):
    Name = 'Model_SDP_Conv_Residual_SingleTel'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for triple telescopes
    First split, and run the input conv on 3 telescopes individually
    Then concatenate the results and run  main conv layers
    '''

    def __init__(self,in_main_channels = (3,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_SDP_Conv_Residual_SingleTel, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout2d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0_large = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        self.conv_0_small = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=(1,0)) # Out=> (N, N_kernels, 20, 22)
        self.conv_0       = nn.Conv2d(N_kernels*2, N_kernels, kernel_size=3, padding=1) # Out=> (N, N_kernels, 20, 22)
        self.BN_0         = nn.BatchNorm2d(N_kernels)
        self.conv_1 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 20, 20)
        self.BN_1   = nn.BatchNorm2d(N_kernels)
        # Pool here to (N, N_kernels, 10, 10)
        self.conv_2 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) # Out=> (N, N_kernels, 10, 10)
        self.BN_2   = nn.BatchNorm2d(N_kernels)
        # Pooling here to (N, N_kernels, 5, 5)
        self.conv_3 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_3   = nn.BatchNorm2d(N_kernels)
        self.pool = nn.MaxPool2d(2,2)

        # Reshape to (N, N_kernels*5*5)
        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*5*5, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.SDPTheta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

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
        Main = self.pool(Main)
        Main = self.Conv_Dropout(self.conv_2(Main))
        Main = self.BN_2(Main)
        Main = self.pool(Main)
        Main = self.conv_3(Main)
        Main = self.BN_3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        Theta = self.Dense_Activation(self.SDPTheta1(Main))
        Theta = self.Dense_Activation(self.SDPTheta2(Theta))
        Theta = self.Angle_Activation(self.SDPTheta3(Theta))

        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        Output = torch.cat([Theta,Phi],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output    
    
class Model_SDP_Conv_Residual_SingleTel_JustTheta(Model_SDP_Conv_Residual_SingleTel):
    Name = 'Model_SDP_Conv_Residual_SingleTel_JustTheta'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_SingleTel_JustTheta, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])

class Model_SDP_Conv_Residual_SingleTel_JustPhi(Model_SDP_Conv_Residual_SingleTel):
    Name = 'Model_SDP_Conv_Residual_SingleTel_JustPhi'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_SingleTel_JustPhi, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])



class Model_SDP_Conv_Residual_SingleTel_NoPool(nn.Module):
    Name = 'Model_SDP_Conv_Residual_SingleTel_NoPool'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for triple telescopes
    First split, and run the input conv on 3 telescopes individually
    Then concatenate the results and run  main conv layers
    This model does not do max pooling
    '''

    def __init__(self,in_main_channels = (3,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_SDP_Conv_Residual_SingleTel_NoPool, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout2d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0_large = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        self.conv_0_small = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=(1,0)) # Out=> (N, N_kernels, 20, 22)
        self.conv_0       = nn.Conv2d(N_kernels*2, N_kernels, kernel_size=3, padding=1) # Out=> (N, N_kernels, 20, 22)
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
        self.SDPTheta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

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

        Theta = self.Dense_Activation(self.SDPTheta1(Main))
        Theta = self.Dense_Activation(self.SDPTheta2(Theta))
        Theta = self.Angle_Activation(self.SDPTheta3(Theta))

        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        Output = torch.cat([Theta,Phi],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output    
    
class Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta(Model_SDP_Conv_Residual_SingleTel_NoPool):
    Name = 'Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_SingleTel_NoPool_JustTheta, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])

class Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi(Model_SDP_Conv_Residual_SingleTel_NoPool):
    Name = 'Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_SingleTel_NoPool_JustPhi, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])




class Model_SDP_Conv_Residual_TripleTel_NoPool(nn.Module):
    Name = 'Model_SDP_Conv_Residual_TripleTel_NoPool'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for triple telescopes
    First split, and run the input conv on 3 telescopes individually
    Then concatenate the results and run  main conv layers
    This model does not do max pooling
    '''

    def __init__(self,in_main_channels = (3,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_SDP_Conv_Residual_TripleTel_NoPool, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout2d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0_large = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        self.conv_0_small = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=(1,0)) # Out=> (N, N_kernels, 20, 22)
        self.conv_0       = nn.Conv2d(N_kernels*2, N_kernels, kernel_size=3, padding=1) # Out=> (N, N_kernels, 20, 22)
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
        self.Dense1 = nn.Linear(N_kernels*60*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.SDPTheta1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])

    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 1, 'Only one Main is expected'
        Main = Main[0].to(device)
        # Split Main into 3 telescopes
        Main_1 = Main[:,:, 0:20,:]
        Main_2 = Main[:,:,20:40,:]
        Main_3 = Main[:,:,40:60,:]
        

        Main_L_1 = self.Conv_Activation(self.conv_0_large(Main_1))
        Main_S_1 = self.Conv_Activation(self.conv_0_small(Main_1))
        
        Main_L_2 = self.Conv_Activation(self.conv_0_large(Main_2))
        Main_S_2 = self.Conv_Activation(self.conv_0_small(Main_2))
        
        Main_L_3 = self.Conv_Activation(self.conv_0_large(Main_3))
        Main_S_3 = self.Conv_Activation(self.conv_0_small(Main_3))
        

        Main_L = torch.cat([Main_L_1,Main_L_2,Main_L_3],dim=2)
        Main_S = torch.cat([Main_S_1,Main_S_2,Main_S_3],dim=2)
        
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

        Theta = self.Dense_Activation(self.SDPTheta1(Main))
        Theta = self.Dense_Activation(self.SDPTheta2(Theta))
        Theta = self.Angle_Activation(self.SDPTheta3(Theta))

        Phi   = self.Dense_Activation(self.SDPPhi1(Main))
        Phi   = self.Dense_Activation(self.SDPPhi2(Phi))
        Phi   = self.Angle_Activation(self.SDPPhi3(Phi))

        Output = torch.cat([Theta,Phi],dim=1)
        Output = Output*self.OutWeights.to(device)

        return Output    
    


class Model_SDP_Conv_Residual_TripleTel_NoPool_JustTheta(Model_SDP_Conv_Residual_TripleTel_NoPool):
    Name = 'Model_SDP_Conv_Residual_TripleTel_NoPool_JustTheta'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Theta is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_TripleTel_NoPool_JustTheta, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])


class Model_SDP_Conv_Residual_TripleTel_NoPool_JustPhi(Model_SDP_Conv_Residual_TripleTel_NoPool):
    Name = 'Model_SDP_Conv_Residual_TripleTel_NoPool_JustPhi'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Only the Phi is learned
    '''

    def __init__(self, in_main_channels=(3,), N_kernels=32, N_dense_nodes=128, **kwargs):
        super(Model_SDP_Conv_Residual_TripleTel_NoPool_JustPhi, self).__init__(in_main_channels=in_main_channels, N_kernels=N_kernels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])


