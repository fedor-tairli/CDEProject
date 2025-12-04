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
    
def Loss(Pred,Truth,keys = ['SDPTheta','SDPPhi'],ReturnTensor = True):
    
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
    Truth_Classes = (torch.abs(RecValues - Truth_RecValues) < 1e-3).float()
    
    # Now, calculate the weights for each guess
    Augmentation_magnitude = torch.abs(RecValues - Truth_RecValues)
    
    weights = torch.ones_like(Truth_Classes)
    label_T = Truth_Classes == 1
    label_F = Truth_Classes == 0 
    guess_T = Pred_Classes >= 0.5
    guess_F = (Pred_Classes <  0.5) & (Pred_Classes >= 0.0)
    
    
    weights[label_T & guess_T] = 1 # Nominal weight # Aug.Mag = 0 here
    weights[label_T & guess_F] = 5 # High weight    # Aug.Mag = 0 here
    weights[label_F & guess_F] = 2 # increased weight # Aug.Mag doesnt matter here
    weights[label_F & guess_T] = Augmentation_magnitude[label_F & guess_T]* 5
    
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
    
def Loss_Reg(Pred,Truth,keys = ['SDPTheta','SDPPhi'],ReturnTensor = True):
    '''
    Mean Squared Error Loss for Regression Values
    Binary cross entropy loss for all predicted values. 
    Assumes that the predicted values are in [0,1] range (Sigmoid Activation)
    Function automatically accounts for augmentations and scales in the model's forward augmentation
    '''
    # First, we expect pred to be a list of [predicted classes, RecValues]
    assert isinstance(Pred,list) or isinstance(Pred, tuple) or isinstance(Pred, dict), 'Predictions should be a list of [PredictedClasses, RecValues]'
    assert len(Pred) == 3, 'Predictions should be a list of [PredictedClasses, Augmented RecValues, Regression RecValues]'

    Pred_Classes = Pred[0]
    RecValues    = Pred[1] # Augmented RecValues
    Reg_Values   = Pred[2] # Regression RecValues

    # Making Sure Devices are the same
    RecValues = RecValues.to(Pred_Classes.device)
    Truth     = Truth    .to(Pred_Classes.device)
    
    # figure out the augmentation scale
    assert RecValues.shape[0] % Truth.shape[0] == 0, f'Prediction and Truth sizes are not compatible, cannot determine augmentation scale {RecValues.shape[0]} vs {Truth.shape[0]}'

    Augmentation_Scale = RecValues.shape[0] // Truth.shape[0]
    Truth_RecValues    = Truth.repeat_interleave(Augmentation_Scale,dim = 0)
    Truth_Classes = (torch.abs(RecValues - Truth_RecValues) < 1e-3).float()
    
    # Now, calculate the weights for each guess
    Augmentation_magnitude = torch.abs(RecValues - Truth_RecValues)
    
    weights = torch.ones_like(Truth_Classes)
    label_T = Truth_Classes == 1
    label_F = Truth_Classes == 0 
    guess_T = Pred_Classes >= 0.5
    guess_F = (Pred_Classes <  0.5) & (Pred_Classes >= 0.0)
    
    
    weights[label_T & guess_T] = 1 # Nominal weight # Aug.Mag = 0 here
    weights[label_T & guess_F] = 5 # High weight    # Aug.Mag = 0 here
    weights[label_F & guess_F] = 2 # increased weight # Aug.Mag doesnt matter here
    weights[label_F & guess_T] = Augmentation_magnitude[label_F & guess_T]* 3
    
    losses = {}
    for i,key in enumerate(keys):
        losses[key] = F.binary_cross_entropy(Pred_Classes[:,i],Truth_Classes[:,i],weight = weights[:,i]) + F.mse_loss(Reg_Values[:,i],Truth[:,i])

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
    

def validate_Reg(model,Dataset,Loss,device,BatchSize = 64):
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
    RegVals = []
    Truths  = []
    
    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth,_  in Dataset:
            Model_out = model(BatchMains,BatchAux,Augmentation_Scale=2)

            Preds  .append(Model_out[0].to('cpu'))
            RecVals.append(Model_out[1].to('cpu'))
            RegVals.append(Model_out[2].to('cpu'))
            Truths .append(BatchTruth  .to('cpu'))
        
        
        Preds   = torch.cat(Preds  ,dim=0)
        RecVals = torch.cat(RecVals,dim=0)
        RegVals = torch.cat(RegVals,dim=0)
        Truths  = torch.cat(Truths ,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)

    return Loss([Preds,RecVals,RegVals],Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)



def metric(model,Dataset,device,keys=['SDPTheta','SDPPhi'],BatchSize = 64,metric_style = 'Accuracy'):
    '''
    Takes model, Dataset,device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    BatchSize to change in case it doesnt fit into memory
    Returns accuracy or other metric
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


    metrics = {'Units':[]}
    
    for i,key in enumerate(keys):
        if metric_style == 'Accuracy':
            correct = (Truth_labels[:,i] == Pred_labels[:,i]).float()
            accuracy = correct.sum() / len(correct)
            metrics[key] = accuracy.item()*100
            metrics['Units'].append('%')
        else:
            raise NotImplementedError(f'Metric Style {metric_style} not implemented')
    
    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)
    return metrics

def metric_Reg(model,Dataset,device,keys=['SDPTheta','SDPPhi'],BatchSize = 64,metric_style = 'Accuracy'):
    '''
    Takes model, Dataset,device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    BatchSize to change in case it doesnt fit into memory
    Returns accuracy or other metric
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    
    Dataset.BatchSize = Dataset.BatchSize*(BatchSize//8)
    Preds   = []
    RecVals = []
    RegVals = []
    Truths  = []

    with torch.no_grad():
        for _, BatchMains, BatchAux, BatchTruth, _ in Dataset:
            Model_out = model(BatchMains,BatchAux)
            Preds  .append(Model_out[0].to('cpu'))
            RecVals.append(Model_out[1].to('cpu'))
            RegVals.append(Model_out[2].to('cpu'))
            Truths .append(BatchTruth  .to('cpu'))

    Preds   = torch.cat(Preds  ,dim=0)
    RecVals = torch.cat(RecVals,dim=0)
    RegVals = torch.cat(RegVals,dim=0)
    Truths  = torch.cat(Truths ,dim=0)
    
    # Augmentation scale 
    Augmentation_Scale = RecVals.shape[0] // Truths.shape[0]
    RegTruths = Truths.clone()
    Truths = Truths.repeat_interleave(Augmentation_Scale,dim = 0)

    Truth_labels = (RecVals == Truths).float()
    Pred_labels = (Preds >= 0.5).float()


    metrics = {'Units':[]}
    
    for i,key in enumerate(keys):
        if metric_style == 'Accuracy':
            correct = (Truth_labels[:,i] == Pred_labels[:,i]).float()
            accuracy = correct.sum() / len(correct)
            metrics[key] = accuracy.item()*100
            metrics['Units'].append('%')
        else:
            raise NotImplementedError(f'Metric Style {metric_style} not implemented')
        AngDiv = torch.atan2(torch.sin(RegVals[:,i]-RegTruths[:,i]),torch.cos(RegVals[:,i]-RegTruths[:,i]))
        metrics[key+'AngDiv'] = torch.quantile(torch.abs(AngDiv),0.68)
        metrics['Units'].append('rad')

    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)
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









class Model_SDP_NLRE_with_Conv_Regression(nn.Module):
    Name = 'Model_SDP_NLRE_with_Conv_Regression'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        
        assert len(in_main_channels) == 2, 'Expecting two Mains: TelInpus and RecValues'
        
        in_RecValues_channels = in_main_channels[1]
        in_main_channels      = in_main_channels[0]
        
        self.in_main_channels      = in_main_channels
        self.in_RecValues_channels = in_RecValues_channels
        
        assert in_RecValues_channels == 2, 'Expecting 2 RecValues Channels'
        self.kwargs = kwargs
        dropout = kwargs['model_Dropout'] if 'model_Dropout' in kwargs else 0.2

        super(Model_SDP_NLRE_with_Conv_Regression, self).__init__()

        # Activation Function
        self.Conv_Activation   = nn.LeakyReLU()
        self.Dense_Activation  = nn.ReLU()
        self.Angle_Activation  = nn.Tanh()
        self.Binary_Activation = nn.Sigmoid()
        self.Conv_Dropout  = nn.Dropout2d(dropout)
        self.Dense_Dropout = nn.Dropout(dropout)
        
        # Graph Convolution Layers # Input should be (N, in_main_channels, 20, 22) for three telescopes
        self.conv_0_large = nn.Conv2d(in_main_channels, N_kernels, kernel_size=5, padding=(2,1)) # Out=> (N, N_kernels, 20, 20)
        self.conv_0_small = nn.Conv2d(in_main_channels, N_kernels, kernel_size=3, padding=(1,0)) # Out=> (N, N_kernels, 20, 22)
        self.conv_0       = nn.Conv2d(N_kernels*2, N_kernels, kernel_size=3, padding=1) # Out=> (N, N_kernels, 20, 22)
        self.BN_0         = nn.BatchNorm2d(N_kernels)
        
        self.conv_1 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) 
        self.BN_1   = nn.BatchNorm2d(N_kernels)

        self.conv_2 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_2   = nn.BatchNorm2d(N_kernels)
        
        self.conv_3 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_3   = nn.BatchNorm2d(N_kernels)

        self.conv_4 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_4   = nn.BatchNorm2d(N_kernels)

        self.conv_5 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.BN_5   = nn.BatchNorm2d(N_kernels)


        # Reshape to (N, N_kernels*20*20)
        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layers
        self.SDPTheta_Reg1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta_Reg2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta_Reg3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPTheta_Cls1 = nn.Linear(N_dense_nodes+4,N_dense_nodes)
        self.SDPTheta_Cls2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta_Cls3 = nn.Linear(N_dense_nodes//2,1)

        self.SDPPhi_Reg1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi_Reg2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPPhi_Reg3   = nn.Linear(N_dense_nodes//2,1)
        
        self.SDPPhi_Cls1   = nn.Linear(N_dense_nodes+4,N_dense_nodes)
        self.SDPPhi_Cls2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi_Cls3   = nn.Linear(N_dense_nodes//2,1)




        self.InWeights  = torch.tensor([1,1])
        self.OutWeights = torch.tensor([1,1])

        self.GaussianScale = kwargs['GaussianScale'] if 'GaussianScale' in kwargs else 5.0 # degrees

    def forward(self,Main,Aux,Augmentation_Scale = None, Augmentation_Function = 'GaussianShift',Augmentation_Magnitude = None):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert len(Main) == 2, 'Two Mains are expected'

        
        if Main[0].ndim == 4: # Main should have 3 dims, RecVals 2 dims , forwards order
            RecVals = Main[1].to(device)
            Main    = Main[0].to(device)
        elif Main[1].ndim == 4: # Backwards order
            RecVals = Main[0].to(device)
            Main    = Main[1].to(device)
        else:
            raise ValueError('Cannot determine Main and RecVals')

        Main_L = self.Conv_Activation(self.conv_0_large(Main))
        Main_S = self.Conv_Activation(self.conv_0_small(Main))
        
        Main = torch.cat([Main_L,Main_S],dim=1)
        Main = self.Conv_Dropout(self.Conv_Activation(self.conv_0(Main)))
        Main = self.BN_0(Main)
    
        Main = self.Conv_Dropout(self.conv_1(Main))
        Main = self.BN_1(Main)
    
        Main = self.Conv_Dropout(self.conv_2(Main))
        Main = self.BN_2(Main)

        Main = self.Conv_Dropout(self.conv_3(Main))
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

        # Regression is done without augmentation
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg1(Main)))
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg2(Theta_Reg)))
        Theta_Reg = self.Angle_Activation(self.SDPTheta_Reg3(Theta_Reg)) * torch.pi # Output in radians between -pi and pi

        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg1(Main)))
        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg2(Phi_Reg)))
        Phi_Reg   = self.Angle_Activation(self.SDPPhi_Reg3(Phi_Reg)) * torch.pi # Output in radians between -pi and pi
        
        Reg_RecVals = torch.cat([Theta_Reg,Phi_Reg],dim=1) * self.OutWeights.to(device) 
        
        # Here Be Augmentation
        if Augmentation_Scale is None:
            Augmentation_Scale = 20
        

        Preds_list             = []
        Augmented_RecVals_list = []



        # Here the model behaviour changes based on wether its training or inference
        if self.training:
            # Augmentation for classification is done around the true rec vals

            for aug_step in range(Augmentation_Scale):
                
                if aug_step ==0: # First one is the original
                    Aug_RecVals = RecVals
                else:

                    if Augmentation_Function == 'BatchShuffle':
                        Aug_RecVals = RecVals[torch.randperm(RecVals.shape[0])]
                    

                    elif Augmentation_Function == 'GaussianShift':
                        gaussian_shift = torch.randn(RecVals.shape[0],RecVals.shape[1]).to(RecVals.device)
                        Aug_RecVals = RecVals + gaussian_shift * self.GaussianScale* torch.pi/180.0
                    
                    else:
                        raise NotImplementedError(f'Augmentation Function {Augmentation_Function} not implemented')

                Aug_RecVals = Aug_RecVals* self.InWeights.to(RecVals.device)

                Trig_Aug_RecVals = torch.cat([torch.sin(Aug_RecVals),torch.cos(Aug_RecVals)],dim=1).to(Main.device)
                Aug_Main = torch.cat([Main,Trig_Aug_RecVals],dim=1)

                Theta = self.Dense_Activation(self.SDPTheta_Cls1(Aug_Main))
                Theta = self.Dense_Activation(self.SDPTheta_Cls2(Theta))
                Theta = self.Binary_Activation(self.SDPTheta_Cls3(Theta))

                Phi   = self.Dense_Activation(self.SDPPhi_Cls1(Aug_Main))
                Phi   = self.Dense_Activation(self.SDPPhi_Cls2(Phi))
                Phi   = self.Binary_Activation(self.SDPPhi_Cls3(Phi))

                These_Preds = torch.cat([Theta,Phi],dim=1) * self.OutWeights.to(device) + (1-self.OutWeights.to(device))*0.5
                Preds_list.append(These_Preds)
                Augmented_RecVals_list.append(Aug_RecVals)

            Preds             = torch.stack(Preds_list            ,dim=0)
            Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=0)
            
            Preds             = Preds            .permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            Augmented_RecVals = Augmented_RecVals.permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            
            return [Preds,Augmented_RecVals,Reg_RecVals]
    
        else:
            # In inference, we do augmentation around the predicted rec vals from regression head
            if Augmentation_Magnitude is None:
                Augmentation_Magnitude = 10*torch.pi/180.0 # 10 degrees in radians
            
            Augmentation_step_size = Augmentation_Magnitude / (Augmentation_Scale//2)

            for aug_step in range(-Augmentation_Scale//2, Augmentation_Scale//2):
                
                Aug_Theta = Theta_Reg + aug_step * Augmentation_step_size
                Aug_Phi   = Phi_Reg   + aug_step * Augmentation_step_size
                Aug_RecVals = torch.cat([Aug_Theta,Aug_Phi],dim=1)
                
                
                Aug_RecVals = Aug_RecVals* self.InWeights.to(RecVals.device)

                Trig_Aug_RecVals = torch.cat([torch.sin(Aug_RecVals),torch.cos(Aug_RecVals)],dim=1).to(Main.device)
                Aug_Main = torch.cat([Main,Trig_Aug_RecVals],dim=1)

                Theta = self.Dense_Activation(self.SDPTheta_Cls1(Aug_Main))
                Theta = self.Dense_Activation(self.SDPTheta_Cls2(Theta))
                Theta = self.Binary_Activation(self.SDPTheta_Cls3(Theta))

                Phi   = self.Dense_Activation(self.SDPPhi_Cls1(Aug_Main))
                Phi   = self.Dense_Activation(self.SDPPhi_Cls2(Phi))
                Phi   = self.Binary_Activation(self.SDPPhi_Cls3(Phi))

                These_Preds = torch.cat([Theta,Phi],dim=1) * self.OutWeights.to(device) + (1-self.OutWeights.to(device))*0.5
                Preds_list.append(These_Preds)
                Augmented_RecVals_list.append(Aug_RecVals)

            Preds             = torch.stack(Preds_list            ,dim=0)
            Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=0)
            
            Preds             = Preds            .permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            Augmented_RecVals = Augmented_RecVals.permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            
            return [Preds,Augmented_RecVals,Reg_RecVals]
                
                
