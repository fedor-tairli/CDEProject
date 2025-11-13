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
    
def Loss(Pred,Truth,keys = ['LogE','Xmax','Chi0','Rp','SDPTheta','SDPPhi'],ReturnTensor = True):
    
    '''
    Binary cross entropy loss for all predicted values. 
    Assumes that the predicted values are in [0,1] range (Sigmoid Activation)
    Function automatically accounts for augmentations and scales in the model's forward augmentation

    '''
    # First, we expect pred to be a list of [predicted classes, RecValues]
    assert isinstance(Pred,list) or isinstance(Pred, tuple) or isinstance(Pred, dict), 'Predictions should be a list of [PredictedClasses, RecValues]'
    
    Pred_Classes = Pred[0]
    RecValues    = Pred[1] # Augmented RecValues
    
    # Making Sure Devices are the same
    RecValues = RecValues.to(Pred_Classes.device)
    Truth     = Truth    .to(Pred_Classes.device)
    
    # figure out the augmentation scale
    assert RecValues.shape[0] % Truth.shape[0] == 0, f'Prediction and Truth sizes are not compatible, cannot determine augmentation scale {RecValues.shape[0]} vs {Truth.shape[0]}'

    Augmentation_Scale = RecValues.shape[0] // Truth.shape[0]
    Truth_RecValues    = Truth.repeat_interleave(Augmentation_Scale,dim = 0)
    Truth_Classes      = (RecValues == Truth_RecValues).float()

    # Now, calculate the weights for each guess\
    Augmentation_magnitude = torch.abs(RecValues - Truth_RecValues)
    weights = torch.ones_like(Truth_Classes)
    label_T = Truth_Classes == 1
    label_F = Truth_Classes == 0 
    guess_T = Pred_Classes >= 0.5
    guess_F = Pred_Classes <  0.5

    weights[label_T & guess_T] = 1 # Nominal weight # Aug.Mag = 0 here
    weights[label_T & guess_F] = 5 # High weight    # Aug.Mag = 0 here
    weights[label_F & guess_F] = 2 # increased weight # Aug.Mag doesnt matter here
    weights[label_F & guess_T] = Augmentation_magnitude[label_F & guess_T] 
    

    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.binary_cross_entropy(Pred_Classes[:,i],Truth_Classes[:,i],weight = weights[:,i])

    # No weighting for now
    # losses = {}
    # for i,key in enumerate(keys):
    #     losses[key] = F.binary_cross_entropy(Pred_Classes[:,i],Truth_Classes[:,i])
    

    losses['Total'] = sum(losses.values())
    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses

def validate(model,Dataset,Loss,device,BatchSize = 64):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    
    Custom validate function to be used for the NLRE model
    NLRE returns a list of [PredictedClasses, RecValues]
    using Augmentation Scale of 2 during validation
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    
    Dataset.BatchSize = Dataset.BatchSize*(BatchSize//8)

    Preds   = []
    RecVals = []
    Truths  = []
    
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth,_  in Dataset:
            Model_out = model(BatchMains,BatchAux,Augmentation_Scale=2)

            Preds  .append(Model_out[0].to('cpu'))
            RecVals.append(Model_out[1].to('cpu'))
            Truths .append(BatchTruth  .to('cpu'))
        
        
        Preds   = torch.cat(Preds  ,dim=0)
        RecVals = torch.cat(RecVals,dim=0)
        Truths  = torch.cat(Truths ,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)

    return Loss([Preds,RecVals],Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)
    

def metric(model,Dataset,device,keys=['Xmax','LogE'],BatchSize = 64,metric_style = 'Accuracy'):
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
    
    Dataset.BatchSize = Dataset.BatchSize*(BatchSize//8)
    Preds   = []
    RecVals = []
    Truths  = []

    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth, _ in Dataset:
            Model_out = model(BatchMains,BatchAux)
            Preds  .append(Model_out[0].to('cpu'))
            RecVals.append(Model_out[1].to('cpu'))
            Truths .append(BatchTruth  .to('cpu'))

    Preds   = torch.cat(Preds  ,dim=0).cpu()
    Truths  = torch.cat(Truths ,dim=0).cpu()
    RecVals = torch.cat(RecVals,dim=0).cpu()
    
    # Augmentation scale 
    Augmentation_Scale = RecVals.shape[0] // Truths.shape[0]
    Truths = Truths.repeat_interleave(Augmentation_Scale,dim = 0)

    Truth_labels = (RecVals == Truths).float()
    Pred_labels = (Preds >= 0.5).float()


    metrics = {}
    for i,key in enumerate(keys):
        if metric_style == 'Accuracy':
            correct = (Truth_labels[:,i] == Pred_labels[:,i]).float()
            accuracy = correct.sum() / len(correct)
            metrics[key] = accuracy.item()
        else:
            raise NotImplementedError(f'Metric Style {metric_style} not implemented')
        
    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)
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


def NLRE_Augmentation_GaussianShift(inputs, outputs, scale, is_training):
    ''' Function to apply the data augmentation to the inputs and outputs
    This function creates copies of the inputs, depending on scale
    For additional copies, it applies a gaussian shift to RecValues(outputs)

    RecValues should be normalised to 0 mean and 1 std beforehand
    
    Output order: [event0, event0_aug1, event0_aug2, ..., event1, event1_aug1, ...]
    '''
    if scale == 1:
        return inputs, outputs
    
    if not is_training:  # Needed for calibration - Ignore scale and add one copy
        scale = 2
    
    NEvents = inputs.shape[0]
    
    # Use repeat_interleave to group augmented copies by event
    input_copies = inputs.repeat_interleave(scale, dim=0)
    output_copies = outputs.repeat_interleave(scale, dim=0)

    # Apply Gaussian shift to augmented copies (skip original copies at indices 0, scale, 2*scale, ...)
    # Indices for the i-th augmented copy of each event
    aug_indices = torch.arange(NEvents, device=outputs.device) * scale # Currently points to original copies
    for i in range(1, scale):
        aug_indices += 1
        # Apply Gaussian shift to the outputs at these indices
        gaussian_shift = torch.randn(NEvents, outputs.shape[1], device=outputs.device)
        output_copies[aug_indices] += gaussian_shift
    
    return input_copies, output_copies



class Model_NLRE_with_Conv3d(nn.Module):

    Name = 'Model_NLRE_with_Conv3d'
    Description = '''
    Convolutional Neural Network which takes in 3d Traces and reconstruction values
    Uses ConvSkip Blocks with Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    Final output is a likelyhood estimation of each reconstruction value being correct

    This model plugs in all (other than the predicted value) reconstruction values into dense layers
    '''



    def __init__(self,in_main_channels = (1,6), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 2, 'Expecting two Main Channels: Traces and RecValues'
        in_RecValues_channels = in_main_channels[1]
        in_main_channels = in_main_channels[0]
        
        self.in_main_channels       = in_main_channels
        self.in_RecValues_channels = in_RecValues_channels
        
        assert in_RecValues_channels == 6, 'Expecting 6 RecValues Channels'
        self.kwargs = kwargs
        dropout = kwargs['model_Dropout'] if 'model_Dropout' in kwargs else 0.2

        super(Model_NLRE_with_Conv3d, self).__init__()

        # Activation Function
        self.Conv_Activation   = nn.LeakyReLU()
        self.Dense_Activation  = nn.ReLU()
        self.Angle_Activation  = nn.Tanh()
        self.Binary_Activation = nn.Sigmoid()
        self.Conv_Dropout      = nn.Dropout3d(dropout)
        self.Dense_Dropout     = nn.Dropout(dropout)


        self.conv0 = nn.Conv3d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,1,0) , stride = 1) # Out=> (N, N_kernels, 40, 20, 20)
        self.Conv1 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv2 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv3 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)

        self.Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.LogE1     = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.LogE2     = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.LogE3     = nn.Linear(N_dense_nodes//2, 1)

        self.Xmax1     = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.Xmax2     = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Xmax3     = nn.Linear(N_dense_nodes//2, 1)

        self.Chi0_1    = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.Chi0_2    = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Chi0_3    = nn.Linear(N_dense_nodes//2, 1)

        self.Rp_1      = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.Rp_2      = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Rp_3      = nn.Linear(N_dense_nodes//2, 1)

        self.SDPTheta1 = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2, 1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes+1, N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2, 1)
        
        self.OutWeights = torch.tensor([1,1,1,1,1,1])


    def forward(self,Graph,Aux=None,Augmentation_Scale = 4,Augmentation_Function = 'GauusianShift'):
        assert Augmentation_Scale > 1, 'Augmentation Scale should be greater than 1'

        # Unpack the Graph Data to Main
        device = self.conv0.weight.device
        NEvents = len(Graph)
        
        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100  ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it. # TODO

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces  = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs      = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys      = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart  = torch.cat(list(map(lambda x : x[3], Graph)))
        
        RecVals = torch.stack(list(map(lambda x : x[4], Graph)),dim=0)

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)

        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1


        # Process the Data - Main Processing is always the same, since main is never augmented
        Main = self.Conv_Activation(self.conv0(Main))
        # Main = self.Conv_Dropout(Main)
        Main = self.Conv1(Main)
        Main = self.Conv2(Main)
        Main = self.Conv3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)

        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense1(Main)))
        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense2(Main)))
        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense3(Main)))

        # Apply Augmentation
        if not self.training:
            Augmentation_Scale = 2

        Preds_list             = []
        Augmented_RecVals_list = []

        for aug_step in range(Augmentation_Scale):
            
            if aug_step ==0: # First one is the original
                Aug_RecVals = RecVals
            else:
                if Augmentation_Function == 'GaussianShift':
                    gaussian_shift = torch.randn(NEvents,RecVals.shape[1]).to(RecVals.device)
                    Aug_RecVals = RecVals + gaussian_shift
                # Other functions to be added here
                elif Augmentation_Function == 'BatchShuffle':
                    perm = torch.randperm(NEvents)
                    Aug_RecVals = RecVals[perm]
                else:
                    raise NotImplementedError(f'Augmentation Function {Augmentation_Function} not implemented')
            
            
        
            # Dense and Output Layers for each value

            # LogE branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,0:1].to(Main.device)],dim=1)
            LogE = self.Dense_Dropout(self.Dense_Activation(self.LogE1(Aug_Main)))
            LogE = self.Dense_Dropout(self.Dense_Activation(self.LogE2(LogE)))
            LogE = self.Binary_Activation(self.LogE3(LogE))

            # Xmax branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,1:2].to(Main.device)],dim=1)
            Xmax = self.Dense_Dropout(self.Dense_Activation(self.Xmax1(Aug_Main)))
            Xmax = self.Dense_Dropout(self.Dense_Activation(self.Xmax2(Xmax)))
            Xmax = self.Binary_Activation(self.Xmax3(Xmax))

            # Chi0 branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,2:3].to(Main.device)],dim=1)
            Chi0 = self.Dense_Dropout(self.Dense_Activation(self.Chi0_1(Aug_Main)))
            Chi0 = self.Dense_Dropout(self.Dense_Activation(self.Chi0_2(Chi0)))
            Chi0 = self.Binary_Activation(self.Chi0_3(Chi0))

            # Rp branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,3:4].to(Main.device)],dim=1)
            Rp = self.Dense_Dropout(self.Dense_Activation(self.Rp_1(Aug_Main)))
            Rp = self.Dense_Dropout(self.Dense_Activation(self.Rp_2(Rp)))
            Rp = self.Binary_Activation(self.Rp_3(Rp))

            # SDPTheta branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,4:5].to(Main.device)],dim=1)
            SDPTheta = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta1(Aug_Main)))
            SDPTheta = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta2(SDPTheta)))
            SDPTheta = self.Binary_Activation(self.SDPTheta3(SDPTheta))

            # SDPPhi branch
            Aug_Main = torch.cat([Main,Aug_RecVals[:,5:6].to(Main.device)],dim=1)
            SDPPhi = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi1(Aug_Main)))
            SDPPhi = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi2(SDPPhi)))
            SDPPhi = self.Binary_Activation(self.SDPPhi3(SDPPhi))

            # Concatenate all reconstruction outputs into final prediction tensor
            # Order: LogE, Xmax, Chi0, Rp, SDPTheta, SDPPhi

            Pred = torch.cat([LogE,Xmax,Chi0,Rp,SDPTheta,SDPPhi],dim=1)

            # Set the output weights -> binary class goes to half for OutWeights == 0
            Pred = Pred * self.OutWeights.to(device)

            # Append to the lists
            Preds_list            .append(Pred)
            Augmented_RecVals_list.append(Aug_RecVals)

        Preds             = torch.stack(Preds_list            ,dim=0)
        Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=0)

        Preds             = Preds            .permute(1,0,2).reshape(-1,self.in_RecValues_channels)
        Augmented_RecVals = Augmented_RecVals.permute(1,0,2).reshape(-1,self.in_RecValues_channels)
        
        return [Preds,Augmented_RecVals]
    


class Model_NLRE_with_Conv3d_BatchShuffle(Model_NLRE_with_Conv3d):

    Name = 'Model_NLRE_with_Conv3d_BatchShuffle'
    Description = '''
    Convolutional Neural Network which takes in 3d Traces and reconstruction values
    Uses ConvSkip Blocks with Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    Final output is a likelyhood estimation of each reconstruction value being correct

    This model plugs in all (other than the predicted value) reconstruction values into dense layers
    Uses Batch Shuffle augmentation
    '''


    def __init__(self,**kwargs):
        super(Model_NLRE_with_Conv3d_BatchShuffle, self).__init__(**kwargs)


    def forward(self,Graph,Aux=None,Augmentation_Scale = 4,Augmentation_Function = 'BatchShuffle'):
        return super().forward(Graph,Aux,Augmentation_Scale,Augmentation_Function)
    





class Model_NLRE_with_Conv3d_AllIn(nn.Module):

    Name = 'Model_NLRE_with_Conv3d_AllIn'
    Description = '''
    Convolutional Neural Network which takes in 3d Traces and reconstruction values
    Uses ConvSkip Blocks with Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    Final output is a likelyhood estimation of each reconstruction value being correct

    This model plugs in all (other than the predicted value) reconstruction values into dense layers
    '''



    def __init__(self,in_main_channels = (1,6), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 2, 'Expecting two Main Channels: Traces and RecValues'
        in_RecValues_channels = in_main_channels[1]
        in_main_channels = in_main_channels[0]
        
        self.in_main_channels       = in_main_channels
        self.in_RecValues_channels = in_RecValues_channels
        
        assert in_RecValues_channels == 6, 'Expecting 6 RecValues Channels'
        self.kwargs = kwargs
        dropout = kwargs['model_Dropout'] if 'model_Dropout' in kwargs else 0.2

        super(Model_NLRE_with_Conv3d_AllIn, self).__init__()

        # Activation Function
        self.Conv_Activation   = nn.LeakyReLU()
        self.Dense_Activation  = nn.ReLU()
        self.Angle_Activation  = nn.Tanh()
        self.Binary_Activation = nn.Sigmoid()
        self.Conv_Dropout      = nn.Dropout3d(dropout)
        self.Dense_Dropout     = nn.Dropout(dropout)


        self.conv0 = nn.Conv3d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,1,0) , stride = 1) # Out=> (N, N_kernels, 40, 20, 20)
        self.Conv1 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv2 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv3 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)

        self.Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.LogE1     = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.LogE2     = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.LogE3     = nn.Linear(N_dense_nodes//2, 1)

        self.Xmax1     = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.Xmax2     = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Xmax3     = nn.Linear(N_dense_nodes//2, 1)

        self.Chi0_1    = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.Chi0_2    = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Chi0_3    = nn.Linear(N_dense_nodes//2, 1)

        self.Rp_1      = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.Rp_2      = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.Rp_3      = nn.Linear(N_dense_nodes//2, 1)

        self.SDPTheta1 = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.SDPTheta2 = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.SDPTheta3 = nn.Linear(N_dense_nodes//2, 1)

        self.SDPPhi1   = nn.Linear(N_dense_nodes+6, N_dense_nodes)
        self.SDPPhi2   = nn.Linear(N_dense_nodes, N_dense_nodes//2)
        self.SDPPhi3   = nn.Linear(N_dense_nodes//2, 1)
        
        self.OutWeights = torch.tensor([1,1,1,1,1,1])


    def forward(self,Graph,Aux=None,Augmentation_Scale = 4,Augmentation_Function = 'GaussianShift'):
        assert Augmentation_Scale > 1, 'Augmentation Scale should be greater than 1'

        # Unpack the Graph Data to Main
        device = self.conv0.weight.device
        NEvents = len(Graph)
        
        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100  ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it. # TODO

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces  = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs      = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys      = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart  = torch.cat(list(map(lambda x : x[3], Graph)))
        
        RecVals = torch.stack(list(map(lambda x : x[4], Graph)),dim=0)

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)

        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1


        # Process the Data - Main Processing is always the same, since main is never augmented
        Main = self.Conv_Activation(self.conv0(Main))
        # Main = self.Conv_Dropout(Main)
        Main = self.Conv1(Main)
        Main = self.Conv2(Main)
        Main = self.Conv3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)

        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense1(Main)))
        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense2(Main)))
        Main = self.Dense_Dropout(self.Dense_Activation(self.Dense3(Main)))

        # Apply Augmentation
        if not self.training:
            Augmentation_Scale = 2

        Preds_list             = []
        Augmented_RecVals_list = []

        for aug_step in range(Augmentation_Scale):
            
            if aug_step ==0: # First one is the original
                Aug_RecVals = RecVals
            else:
                if Augmentation_Function == 'GaussianShift':
                    gaussian_shift = torch.randn(NEvents,RecVals.shape[1]).to(RecVals.device)
                    Aug_RecVals = RecVals + gaussian_shift
                # Other functions to be added here
                elif Augmentation_Function == 'BatchShuffle':
                    perm = torch.randperm(NEvents)
                    Aug_RecVals = RecVals[perm]
                else:
                    raise NotImplementedError(f'Augmentation Function {Augmentation_Function} not implemented')
            
            
        
            # Dense and Output Layers for each value

            # LogE branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            LogE = self.Dense_Dropout(self.Dense_Activation(self.LogE1(Aug_Main)))
            LogE = self.Dense_Dropout(self.Dense_Activation(self.LogE2(LogE)))
            LogE = self.Binary_Activation(self.LogE3(LogE))

            # Xmax branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            Xmax = self.Dense_Dropout(self.Dense_Activation(self.Xmax1(Aug_Main)))
            Xmax = self.Dense_Dropout(self.Dense_Activation(self.Xmax2(Xmax)))
            Xmax = self.Binary_Activation(self.Xmax3(Xmax))

            # Chi0 branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            Chi0 = self.Dense_Dropout(self.Dense_Activation(self.Chi0_1(Aug_Main)))
            Chi0 = self.Dense_Dropout(self.Dense_Activation(self.Chi0_2(Chi0)))
            Chi0 = self.Binary_Activation(self.Chi0_3(Chi0))

            # Rp branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            Rp = self.Dense_Dropout(self.Dense_Activation(self.Rp_1(Aug_Main)))
            Rp = self.Dense_Dropout(self.Dense_Activation(self.Rp_2(Rp)))
            Rp = self.Binary_Activation(self.Rp_3(Rp))

            # SDPTheta branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            SDPTheta = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta1(Aug_Main)))
            SDPTheta = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta2(SDPTheta)))
            SDPTheta = self.Binary_Activation(self.SDPTheta3(SDPTheta))

            # SDPPhi branch
            Aug_Main = torch.cat([Main,Aug_RecVals.to(Main.device)],dim=1)
            SDPPhi = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi1(Aug_Main)))
            SDPPhi = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi2(SDPPhi)))
            SDPPhi = self.Binary_Activation(self.SDPPhi3(SDPPhi))

            # Concatenate all reconstruction outputs into final prediction tensor
            # Order: LogE, Xmax, Chi0, Rp, SDPTheta, SDPPhi

            Pred = torch.cat([LogE,Xmax,Chi0,Rp,SDPTheta,SDPPhi],dim=1)

            # Set the output weights -> binary class goes to half for OutWeights == 0
            Pred = Pred * self.OutWeights.to(device) 

            # Append to the lists
            Preds_list            .append(Pred)
            Augmented_RecVals_list.append(Aug_RecVals)

        Preds             = torch.stack(Preds_list            ,dim=0)
        Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=0)

        Preds             = Preds            .permute(1,0,2).reshape(-1,self.in_RecValues_channels)
        Augmented_RecVals = Augmented_RecVals.permute(1,0,2).reshape(-1,self.in_RecValues_channels)
        
        return [Preds,Augmented_RecVals]
    
class Model_NLRE_with_Conv3d_AllIn_BatchShuffle(Model_NLRE_with_Conv3d_AllIn):
    Name = 'Model_NLRE_with_Conv3d_AllIn_BatchShuffle'
    Description = '''
    Convolutional Neural Network which takes in 3d Traces and reconstruction values
    Uses ConvSkip Blocks with Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    Final output is a likelyhood estimation of each reconstruction value being correct

    This model plugs in all (other than the predicted value) reconstruction values into dense layers
    Uses Batch Shuffle augmentation
    '''


    def __init__(self, in_main_channels = (1,6), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        super(Model_NLRE_with_Conv3d_AllIn_BatchShuffle, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)


    def forward(self,Graph,Aux=None,Augmentation_Scale = 4,Augmentation_Function = 'BatchShuffle'):
        return super().forward(Graph,Aux,Augmentation_Scale,Augmentation_Function)
    



class Model_NLRE_with_Conv3d_AllIn_BatchShuffle_SDPOnly(Model_NLRE_with_Conv3d_AllIn):
    Name = 'Model_NLRE_with_Conv3d_AllIn_BatchShuffle_SDPOnly'
    Description = '''
    Convolutional Neural Network which takes in 3d Traces and reconstruction values
    Uses ConvSkip Blocks with Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    Final output is a likelyhood estimation of each reconstruction value being correct

    This model plugs in all (other than the predicted value) reconstruction values into dense layers
    Uses Batch Shuffle augmentation
    Only predicts SDPTheta and SDPPhi
    '''

    def __init__(self,in_main_channels = (1,6), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        super(Model_NLRE_with_Conv3d_AllIn_BatchShuffle_SDPOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        # Override the output layers to only have SDPTheta and SDPPhi
        self.OutWeights = torch.tensor([0,0,0,0,1,1])

    def forward(self,Graph,Aux=None,Augmentation_Scale = 4,Augmentation_Function = 'BatchShuffle'):
        return super().forward(Graph,Aux,Augmentation_Scale,Augmentation_Function)