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
    
def Loss(Pred,Truth,keys=['Chi0','Rp'],ReturnTensor = True):

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
    

def metric(model,Dataset,device,keys=['Chi0','Rp'],BatchSize = 1024): # I can afford big batch - network is light
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
    
    # Check for SDP_double definition, 
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


class Model_OnePix_Linear(nn.Module):
    Name = 'Model_OnePix_Linear'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Two Main inputs - Trace and Chi_i for each pixel
    '''


    def __init__(self,in_main_channels = (120,1), N_dense_nodes = 128, **kwargs):
        super(Model_OnePix_Linear, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Dense layers for trace (To be replaced with LSTM or something)
        self.Dense1 = nn.Linear(in_main_channels[0], N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)

        self.Rp_1   = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])


    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 2 mains are expected
        assert len(Main) == 2, 'Two Mains are expected'
        Trace = Main[0].to(device)
        Chi_i = Main[1].to(device).unsqueeze(1) # Add a channel dimension

        Trace = self.Dense_Activation(self.Dense1(Trace))
        Trace = self.Dense_Activation(self.Dense2(Trace))
        Trace = self.Dense_Activation(self.Dense3(Trace))

        Chi0 = self.Dense_Activation(self.Chi0_1(torch.cat([Trace,Chi_i],dim=1)))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))

        Rp   = self.Dense_Activation(self.Rp_1(torch.cat([Trace,Chi_i],dim=1)))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        Output = torch.cat([Chi0,Rp],dim=1)

        return Output**self.OutWeights.to(device)
    

class Model_OnePix_Linear_JustChi0(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_JustChi0'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Only the Chi0 is learned
    '''

    def __init__(self, in_main_channels=(1, 1), N_dense_nodes=128, **kwargs):
        super(Model_OnePix_Linear_JustChi0, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([1, 0])

class Model_OnePix_Linear_JustRp(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_JustRp'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Only the Rp is learned
    '''

    def __init__(self, in_main_channels=(1, 1), N_dense_nodes=128, **kwargs):
        super(Model_OnePix_Linear_JustRp, self).__init__(in_main_channels=in_main_channels, N_dense_nodes=N_dense_nodes, **kwargs)
        self.OutWeights = torch.tensor([0, 1])




class Model_OnePix_Linear_NoChii(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_NoChii'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Two Main inputs - Trace and Chi_i for each pixel
    Zero-Out Chii on input
    '''


    def __init__(self,in_main_channels = (120,1), N_dense_nodes = 128, **kwargs):
        super(Model_OnePix_Linear, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Dense layers for trace (To be replaced with LSTM or something)
        self.Dense1 = nn.Linear(in_main_channels[0], N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)

        self.Rp_1   = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])


    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 2 mains are expected
        assert len(Main) == 2, 'Two Mains are expected'
        Trace = Main[0].to(device)
        Chi_i = Main[1].to(device).unsqueeze(1)*0 # Add a channel dimension

        Trace = self.Dense_Activation(self.Dense1(Trace))
        Trace = self.Dense_Activation(self.Dense2(Trace))
        Trace = self.Dense_Activation(self.Dense3(Trace))

        Chi0 = self.Dense_Activation(self.Chi0_1(torch.cat([Trace,Chi_i],dim=1)))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))

        Rp   = self.Dense_Activation(self.Rp_1(torch.cat([Trace,Chi_i],dim=1)))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        Output = torch.cat([Chi0,Rp],dim=1)

        return Output**self.OutWeights.to(device)



class Model_OnePix_Linear_NoChii_NormTrace(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_NoChii_NormTrace'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Two Main inputs - Trace and Chi_i for each pixel
    Zero-Out Chii on input
    '''


    def __init__(self,in_main_channels = (120,1), N_dense_nodes = 128, **kwargs):
        super(Model_OnePix_Linear_NoChii_NormTrace, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Dense layers for trace (To be replaced with LSTM or something)
        self.Dense1 = nn.Linear(in_main_channels[0], N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)

        self.Rp_1   = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])


    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 2 mains are expected
        assert len(Main) == 2, 'Two Mains are expected'
        Trace = Main[0].to(device)
        Chi_i = Main[1].to(device).unsqueeze(1)*0 # Add a channel dimension

        Trace = Trace / torch.max(Trace,dim=1,keepdim=True)[0] # Normalise the trace
        Trace = self.Dense_Activation(self.Dense1(Trace))
        Trace = self.Dense_Activation(self.Dense2(Trace))
        Trace = self.Dense_Activation(self.Dense3(Trace))

        Chi0 = self.Dense_Activation(self.Chi0_1(torch.cat([Trace,Chi_i],dim=1)))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))

        Rp   = self.Dense_Activation(self.Rp_1(torch.cat([Trace,Chi_i],dim=1)))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        Output = torch.cat([Chi0,Rp],dim=1)

        return Output**self.OutWeights.to(device)



class Model_OnePix_Linear_NoTrace(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_NoTrace'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Two Main inputs - Trace and Chi_i for each pixel
    Zero-Out Trace on input
    '''


    def __init__(self,in_main_channels = (120,1), N_dense_nodes = 128, **kwargs):
        super(Model_OnePix_Linear_NoTrace, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Dense layers for trace (To be replaced with LSTM or something)
        self.Dense1 = nn.Linear(in_main_channels[0], N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)

        self.Rp_1   = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])


    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 2 mains are expected
        assert len(Main) == 2, 'Two Mains are expected'
        Trace = Main[0].to(device)*0
        Chi_i = Main[1].to(device).unsqueeze(1) # Add a channel dimension

        Trace = self.Dense_Activation(self.Dense1(Trace))
        Trace = self.Dense_Activation(self.Dense2(Trace))
        Trace = self.Dense_Activation(self.Dense3(Trace))

        Chi0 = self.Dense_Activation(self.Chi0_1(torch.cat([Trace,Chi_i],dim=1)))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))

        Rp   = self.Dense_Activation(self.Rp_1(torch.cat([Trace,Chi_i],dim=1)))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        Output = torch.cat([Chi0,Rp],dim=1)

        return Output**self.OutWeights.to(device)



class Model_OnePix_Linear_NoTraceNoChii(Model_OnePix_Linear):
    Name = 'Model_OnePix_Linear_NoTraceNoChii'
    Description = '''
    Dense Neural Network for Geometry Reconstruction
    Using Dense Layers we are testing if there is any info in the trace of the pixel by itself
    Two Main inputs - Trace and Chi_i for each pixel
    Zero-Out Trace and Chii on input
    '''


    def __init__(self,in_main_channels = (120,1), N_dense_nodes = 128, **kwargs):
        super(Model_OnePix_Linear_NoTraceNoChii, self).__init__()

        # Activation Function
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        
        # Dense layers for trace (To be replaced with LSTM or something)
        self.Dense1 = nn.Linear(in_main_channels[0], N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layer
        self.Chi0_1 = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Chi0_2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Chi0_3 = nn.Linear(N_dense_nodes//2,1)

        self.Rp_1   = nn.Linear(N_dense_nodes+in_main_channels[1],N_dense_nodes)
        self.Rp_2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Rp_3   = nn.Linear(N_dense_nodes//2,1)

        self.OutWeights = torch.tensor([1,1])


    def forward(self,Main,Aux):
        device = self.Dense1.weight.device
        # Only 2 mains are expected
        assert len(Main) == 2, 'Two Mains are expected'
        Trace = Main[0].to(device)*0
        Chi_i = Main[1].to(device).unsqueeze(1)*0 # Add a channel dimension

        Trace = self.Dense_Activation(self.Dense1(Trace))
        Trace = self.Dense_Activation(self.Dense2(Trace))
        Trace = self.Dense_Activation(self.Dense3(Trace))

        Chi0 = self.Dense_Activation(self.Chi0_1(torch.cat([Trace,Chi_i],dim=1)))
        Chi0 = self.Dense_Activation(self.Chi0_2(Chi0))
        Chi0 = self.Angle_Activation(self.Chi0_3(Chi0))

        Rp   = self.Dense_Activation(self.Rp_1(torch.cat([Trace,Chi_i],dim=1)))
        Rp   = self.Dense_Activation(self.Rp_2(Rp))
        Rp   = self.Rp_3(Rp)

        Output = torch.cat([Chi0,Rp],dim=1)

        return Output**self.OutWeights.to(device)





class Model_SDP_Conv(nn.Module):
    Name = 'Model_SDP_Conv'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers
    Reconstruction is done for one telescope
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
    