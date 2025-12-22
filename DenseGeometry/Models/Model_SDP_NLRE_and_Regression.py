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

# Old - BCE loss
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
    
    Train_Style = 'Both'
    
    if Pred_Classes is None:   Train_Style = 'Regression'
    if Reg_Values   is None:   Train_Style = 'Classification'
    
    # Making Sure Devices are the same
    Device = Pred_Classes.device if Train_Style != 'Regression' else Reg_Values.device
    if Train_Style in ['Classification','Both']:
        Pred_Classes = Pred_Classes.to(Device)
        RecValues    = RecValues   .to(Device)
    if Train_Style in ['Regression','Both']:
        Reg_Values   = Reg_Values  .to(Device)

    Truth     = Truth    .to(Device)
    
    
    # figure out the augmentation scale
    if Train_Style in ['Classification','Both']:
        assert RecValues.shape[0] % Truth.shape[0] == 0, f'Prediction and Truth sizes are not compatible, cannot determine augmentation scale {RecValues.shape[0]} vs {Truth.shape[0]}'

        Augmentation_Scale = RecValues.shape[0] // Truth.shape[0]
        Truth_RecValues    = Truth.repeat_interleave(Augmentation_Scale,dim = 0)
        step_size = RecValues[1,0] - RecValues[0,0] # assuming uniform steps
        truth_gap = step_size * torch.sqrt(torch.tensor(2.0))
        Truth_Classes = (torch.abs(RecValues - Truth_RecValues) < truth_gap).float() # within 1 degrees which is sqrt(step) gap
        
        # Checking for rehaping correctness
        # print(f'Rec Values Shape {RecValues.shape}')

        # for i in range(Truth_RecValues.shape[0]):
        #     print(f'Truth Rec: {Truth_RecValues[i,0].item()*180/np.pi:.2f} , Pred Rec: {RecValues[i,0].item()*180/np.pi:.2f} , Diff: {(RecValues[i,0]-Truth_RecValues[i,0]).item()*180/np.pi:.2f} deg , Class: {Truth_Classes[i,0].item()}')
        
        # raise ValueError('Debug Stop')
        # Now, calculate the weights for each guess

        # Augmentation_magnitude = torch.abs(RecValues - Truth_RecValues)
        
        weights = torch.ones_like(Truth_Classes)
        label_T = Truth_Classes == 1
        label_F = Truth_Classes == 0 

        # Rehsape weights into (BatchSize, Augmentation_Scale, N_Classes)
        weights = weights.view(-1,Augmentation_Scale,Truth_Classes.shape[1])
        label_F = label_F.view(-1,Augmentation_Scale,Truth_Classes.shape[1])
        label_T = label_T.view(-1,Augmentation_Scale,Truth_Classes.shape[1])

        weights[label_T] = Augmentation_Scale * 5 
        weights[label_F] = 1

        weights *= label_T.any(dim=1,keepdim=True).float() # If no true label in the augmentation set, set all weights to zero
        

        weights = weights.detach()

        weights = weights.view(-1,Truth_Classes.shape[1])
        
        


        
        
        
    
    if Train_Style in ['Regression','Both']:
        Truth_Theta_Sin = torch.sin(Truth[:,0]).unsqueeze(1)
        Truth_Theta_Cos = torch.cos(Truth[:,0]).unsqueeze(1)
        Truth_Phi_Sin   = torch.sin(Truth[:,1]).unsqueeze(1)
        Truth_Phi_Cos   = torch.cos(Truth[:,1]).unsqueeze(1)

        Truth = torch.cat([Truth_Theta_Sin,Truth_Theta_Cos,Truth_Phi_Sin,Truth_Phi_Cos],dim=1)

    
    losses = {}
    for i,key in enumerate(keys):
        if Train_Style in ['Regression','Both']:
            if 'SDPTheta' in key:
                losses[key+'_Reg'] = F.mse_loss(Reg_Values[:,0:2],Truth[:,0:2])
            elif 'SDPPhi' in key:
                losses[key+'_Reg'] = F.mse_loss(Reg_Values[:,2:4],Truth[:,2:4])
            else:
                raise ValueError(f'Key {key} not recognized for Regression Loss')
        else:
            losses[key+'_Reg'] = torch.zeros(1).to(Device)
        
        if Train_Style in ['Classification','Both']:
            losses[key+'_Cls'] = F.binary_cross_entropy(Pred_Classes[:,i],Truth_Classes[:,i],weight=weights[:,i])
        else:
            losses[key+'_Cls'] = torch.zeros(1).to(Device)

        
        losses[key] = losses[key+'_Reg'] + losses[key+'_Cls']
        

    losses['Total'] = sum([losses[key] for key in keys])
    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses



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
    
    TrainType = None

    with torch.no_grad():
        for i,(_, BatchMains, BatchAux, BatchTruth,_)  in enumerate(Dataset):
            Model_out = model(BatchMains,BatchAux)
            if i == 0:
                if   Model_out[2] is None: TrainType = 'Classification'
                elif Model_out[0] is None: TrainType = 'Regression'
                else:                      TrainType = 'Both'
            
            if TrainType in ['Classification','Both']:
                Preds  .append(Model_out[0].to('cpu'))
                RecVals.append(Model_out[1].to('cpu'))
            
            if TrainType in ['Regression','Both']:
                RegVals.append(Model_out[2].to('cpu'))
            
            Truths .append(BatchTruth  .to('cpu'))
        
        if TrainType in ['Classification','Both']:
            Preds   = torch.cat(Preds  ,dim=0)
            RecVals = torch.cat(RecVals,dim=0)
        else:
            Preds   = None
            RecVals = None
        
        if TrainType in ['Regression','Both']:
            RegVals = torch.cat(RegVals,dim=0)
        else:
            RegVals = None

        Truths  = torch.cat(Truths ,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = Dataset.BatchSize//(BatchSize//8)

    return Loss([Preds,RecVals,RegVals],Truths,keys=Dataset.Truth_Keys,ReturnTensor=False)


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
    TrainType = None

    with torch.no_grad():
        for i,(_, BatchMains, BatchAux, BatchTruth, _) in enumerate(Dataset):
            Model_out = model(BatchMains,BatchAux)
            if i == 0: # Can check what sort of model we are training
                if   Model_out[2] is None: TrainType = 'Classification'
                elif Model_out[0] is None: TrainType = 'Regression'
                else:                      TrainType = 'Both'

            if TrainType in ['Classification','Both']:
                Preds  .append(Model_out[0].to('cpu'))
                RecVals.append(Model_out[1].to('cpu'))
                
            if TrainType in ['Regression','Both']:
                RegVals.append(Model_out[2].to('cpu'))
            
            Truths .append(BatchTruth  .to('cpu'))

    Truths  = torch.cat(Truths ,dim=0)
    RegTruths = Truths.clone()

    if TrainType in ['Classification','Both']:
        Preds   = torch.cat(Preds  ,dim=0)
        RecVals = torch.cat(RecVals,dim=0)
        # Augmentation scale
        Augmentation_Scale = RecVals.shape[0] // Truths.shape[0]
        Truths = Truths.repeat_interleave(Augmentation_Scale,dim = 0)

        Truth_labels = (RecVals == Truths).float()
        Pred_labels = (Preds >= 0.5).float()

    if TrainType in ['Regression','Both']:
        RegVals = torch.cat(RegVals,dim=0)
        RegVals_Theta = torch.atan2(RegVals[:,0],RegVals[:,1]).unsqueeze(1)
        RegVals_Phi   = torch.atan2(RegVals[:,2],RegVals[:,3]).unsqueeze(1)
        RegVals = torch.cat([RegVals_Theta,RegVals_Phi],dim=1)

    metrics = {'Units':[]}
    
    for i,key in enumerate(keys):
        if TrainType in ['Classification','Both']:
            if metric_style == 'Accuracy':
                correct = (Truth_labels[:,i] == Pred_labels[:,i]).float()
                accuracy = correct.sum() / len(correct)
                metrics[key] = accuracy.item()*100
                metrics['Units'].append('%')
            else:
                raise NotImplementedError(f'Metric Style {metric_style} not implemented')
        if TrainType in ['Regression','Both']:
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


class Model_SDP_NLRE_and_Regression(nn.Module):
    Name = 'Model_SDP_NLRE_and_Regression'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        
        assert len(in_main_channels) == 2, 'Expecting two Mains: TelInpus and RecValues'
        
        in_RecValues_channels = in_main_channels[1]
        in_main_channels      = in_main_channels[0]
        
        self.in_main_channels      = in_main_channels
        self.in_RecValues_channels = in_RecValues_channels
        
        assert in_RecValues_channels == 2, 'Expecting 2 RecValues Channels'
        self.kwargs = kwargs
        dropout = kwargs['model_Dropout'] if 'model_Dropout' in kwargs else 0.2

        super(Model_SDP_NLRE_and_Regression, self).__init__()

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

        self.conv_1 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation) 
        self.conv_2 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.conv_3 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.conv_4 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)
        self.conv_5 = Conv_Skip_Block(N_kernels, N_kernels, self.Conv_Activation)

        self.BN_0   = nn.BatchNorm2d(N_kernels)
        self.BN_1   = nn.BatchNorm2d(N_kernels)
        self.BN_2   = nn.BatchNorm2d(N_kernels)
        self.BN_3   = nn.BatchNorm2d(N_kernels)
        self.BN_4   = nn.BatchNorm2d(N_kernels)
        self.BN_5   = nn.BatchNorm2d(N_kernels)


        # Reshape to (N, N_kernels*20*20)
        # Dense Layers
        self.Dense1 = nn.Linear(N_kernels*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        # Output Layers
        self.SDPTheta_Reg1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPTheta_Reg2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta_Reg3 = nn.Linear(N_dense_nodes//2,2) # Predicsts Theta_c and Theta_s

        self.SDPTheta_Cls1 = nn.Linear(N_dense_nodes+6,N_dense_nodes)
        self.SDPTheta_Cls2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPTheta_Cls3 = nn.Linear(N_dense_nodes//2,1) # Predicsts Likelihood of Theta_c and Theta_s

        self.SDPPhi_Reg1   = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.SDPPhi_Reg2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.SDPPhi_Reg3   = nn.Linear(N_dense_nodes//2,2) # Predicts Phi ...
        
        self.SDPPhi_Cls1   = nn.Linear(N_dense_nodes+6,N_dense_nodes)
        self.SDPPhi_Cls2   = nn.Linear(N_dense_nodes,N_dense_nodes//2)  
        self.SDPPhi_Cls3   = nn.Linear(N_dense_nodes//2,1) # Predicts Likelihood of Phi ...




        self.InWeights  = torch.tensor([1,1])
        self.OutWeights = torch.tensor([1,1])
        self.OutputMode = 'Both' # Both Regression and Classification # Options are 'Both','Regression','Classification' 

        self.GaussianScale = kwargs['GaussianScale'] if 'GaussianScale' in kwargs else 5.0 # degrees

    def Freeze_Regression_Block(self):
        for param in self.conv_0_large.parameters(): param.requires_grad = False
        for param in self.conv_0_small.parameters(): param.requires_grad = False
        for param in self.conv_0.parameters(): param.requires_grad = False
        for param in self.conv_1.parameters(): param.requires_grad = False
        for param in self.conv_2.parameters(): param.requires_grad = False
        for param in self.conv_3.parameters(): param.requires_grad = False
        for param in self.conv_4.parameters(): param.requires_grad = False
        for param in self.conv_5.parameters(): param.requires_grad = False
        for param in self.Dense1.parameters(): param.requires_grad = False
        for param in self.Dense2.parameters(): param.requires_grad = False
        for param in self.Dense3.parameters(): param.requires_grad = False
        for param in self.SDPTheta_Reg1.parameters(): param.requires_grad = False
        for param in self.SDPTheta_Reg2.parameters(): param.requires_grad = False
        for param in self.SDPTheta_Reg3.parameters(): param.requires_grad = False
        for param in self.SDPPhi_Reg1.parameters(): param.requires_grad = False
        for param in self.SDPPhi_Reg2.parameters(): param.requires_grad = False
        for param in self.SDPPhi_Reg3.parameters(): param.requires_grad = False



    def forward(self,Main,Aux,Augmentation_Scale = None, Augmentation_Function = 'UniformNearSample',Augmentation_Magnitude = None):
        device = self.Dense1.weight.device
        # Only 1 main is expected
        assert self.OutputMode in ['Both','Regression','Classification'], f'Output Mode {self.OutputMode} not recognized'
        if self.OutputMode != 'Regression':
            assert self.InWeights.sum() == 1, 'This Network is designed to reconstruct only one of Theta or Phi, Not Both. Set InWeights accordingly'
            assert self.OutWeights.sum() == 1, 'This Network is designed to reconstruct only one of Theta or Phi, Not Both. Set OutWeights accordingly'

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
        
    
        Main = self.BN_1(self.Conv_Dropout(self.conv_1(Main)))
        Main = self.BN_2(self.Conv_Dropout(self.conv_2(Main)))
        Main = self.BN_3(self.Conv_Dropout(self.conv_3(Main)))
        Main = self.BN_4(self.Conv_Dropout(self.conv_4(Main)))
        Main = self.BN_5(self.Conv_Dropout(self.conv_5(Main)))

    
        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))

        # Regression is done without augmentation
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg1(Main)))
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg2(Theta_Reg)))
        Theta_Reg = self.Angle_Activation(self.SDPTheta_Reg3(Theta_Reg))
        # Theta_Reg = self.SDPTheta_Reg3(Theta_Reg)

        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg1(Main)))
        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg2(Phi_Reg)))
        Phi_Reg   = self.Angle_Activation(self.SDPPhi_Reg3(Phi_Reg)) 
        # Phi_Reg   = self.SDPPhi_Reg3(Phi_Reg)

        Reg_RecVals = torch.cat([Theta_Reg * self.OutWeights[0].to(device),
                                 Phi_Reg   * self.OutWeights[1].to(device)
                                 ],dim=1) 
        

        if self.OutputMode == 'Regression':
            return [None,None,Reg_RecVals]

        

        # Here Be Augmentation
        if Augmentation_Scale is None:
            Augmentation_Scale = 26
        
        if (Augmentation_Magnitude is None) and (Augmentation_Function == 'UniformNearSample'):
            Augmentation_Magnitude =  2*torch.pi/180.0 # 2 degrees in radians

        Preds_list             = []
        Augmented_RecVals_list = []

        Regression_Expectation_Theta = Theta_Reg.detach()
        Regression_Expectation_Theta = torch.atan2(Theta_Reg[:,0],Theta_Reg[:,1]).unsqueeze(1)
        Regression_Expectation_Phi   = Phi_Reg  .detach()
        Regression_Expectation_Phi   = torch.atan2(Phi_Reg  [:,0],Phi_Reg  [:,1]).unsqueeze(1)

        

        # Here the model behaviour changes based on wether its training or inference
        if Augmentation_Function != 'NoAugmentation':
            # Augmentation for classification is done around the true rec vals

            if Augmentation_Function == 'UniformNearSample':
                for aug_step in range(-Augmentation_Scale//2, Augmentation_Scale//2):
                    Aug_Rec_Theta = Regression_Expectation_Theta + aug_step * Augmentation_Magnitude * self.InWeights[0]
                    Aug_Rec_Phi   = Regression_Expectation_Phi   + aug_step * Augmentation_Magnitude * self.InWeights[1]
                
                    These_Aug_RecVals = torch.cat([Aug_Rec_Theta,Aug_Rec_Phi],dim=1)
                    Augmented_RecVals_list.append(These_Aug_RecVals)
                
                    # Aug Rec Vals Are the :
                    # - Augmented Reconstruction Value Theta/Phi, depending on the model we are Training
                    # - And the Regression Prediction for the Other value
                    
                    if self.OutWeights[0] == 1:
                        Aug_Rec_Theta = torch.cat([torch.sin(Aug_Rec_Theta),torch.cos(Aug_Rec_Theta)],dim=1)
                        Theta_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Aug_Rec_Theta],dim=1)
                        
                        Theta = self.Dense_Activation(self.SDPTheta_Cls1(Theta_Cls_Inp))
                        Theta = self.Dense_Activation(self.SDPTheta_Cls2(Theta))
                        Theta = self.Binary_Activation(self.SDPTheta_Cls3(Theta))
                    else:
                        Theta = torch.tensor([[0.5]]).repeat(Main.shape[0],1).to(device)

                    if self.OutWeights[1] == 1:
                        Aug_Rec_Phi = torch.cat([torch.sin(Aug_Rec_Phi),torch.cos(Aug_Rec_Phi)],dim=1)
                        Phi_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Aug_Rec_Phi],dim=1)

                        Phi = self.Dense_Activation(self.SDPPhi_Cls1(Phi_Cls_Inp))
                        Phi = self.Dense_Activation(self.SDPPhi_Cls2(Phi))
                        Phi = self.Binary_Activation(self.SDPPhi_Cls3(Phi))
                    else:
                        Phi = torch.tensor([[0.5]]).repeat(Main.shape[0],1).to(device)
                    

                    These_Preds = torch.cat([Theta,Phi],dim=1)
                    Preds_list.append(These_Preds)
            else:
                raise NotImplementedError(f'Augmentation Function {Augmentation_Function} not implemented')
            
            Preds             = torch.stack(Preds_list            ,dim=0)
            Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=0)
            
            Preds             = Preds            .permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            Augmented_RecVals = Augmented_RecVals.permute(1,0,2).reshape(-1,self.in_RecValues_channels)
            
            if self.OutputMode == 'Classification':
                return [Preds,Augmented_RecVals,None]
            if self.OutputMode == 'Both':
                return [Preds,Augmented_RecVals,Reg_RecVals]

    
        elif Augmentation_Function == 'NoAugmentation':
            # No Augmentation, Use the RecVals provided
            Rec_Theta = RecVals[:,0].unsqueeze(1)
            Rec_Phi   = RecVals[:,1].unsqueeze(1)

            if self.OutWeights[0] == 1:
                Rec_Theta_in = torch.cat([torch.sin(Rec_Theta),torch.cos(Rec_Theta)],dim=1)
                Theta_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Rec_Theta_in],dim=1)
                
                Theta = self.Dense_Activation(self.SDPTheta_Cls1(Theta_Cls_Inp))
                Theta = self.Dense_Activation(self.SDPTheta_Cls2(Theta))
                Theta = self.Binary_Activation(self.SDPTheta_Cls3(Theta))
            else:
                Theta = torch.tensor([[0.5]]).repeat(Main.shape[0],1).to(device)

            if self.OutWeights[1] == 1:
                Rec_Phi_in = torch.cat([torch.sin(Rec_Phi),torch.cos(Rec_Phi)],dim=1)
                Phi_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Rec_Phi_in],dim=1)

                Phi = self.Dense_Activation(self.SDPPhi_Cls1(Phi_Cls_Inp))
                Phi = self.Dense_Activation(self.SDPPhi_Cls2(Phi))
                Phi = self.Binary_Activation(self.SDPPhi_Cls3(Phi))
            else:
                Phi = torch.tensor([[0.5]]).repeat(Main.shape[0],1).to(device)
            
            These_Preds = torch.cat([Theta,Phi],dim=1)
            if self.OutputMode == 'Classification':
                return [These_Preds,RecVals,None]
            if self.OutputMode == 'Both':
                return [These_Preds,RecVals,Reg_RecVals]


class Model_SDP_NLRE_and_Regression_RegressionOnly(Model_SDP_NLRE_and_Regression):
    Name = 'Model_SDP_NLRE_and_Regression_RegressionOnly'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Does not output likelihoods, only regression values
    --
    Designed as pre-training model for the full NLRE + Regression Model
    Normal init and forward
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        super(Model_SDP_NLRE_and_Regression_RegressionOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        self.OutputMode = 'Regression'


class Model_SDP_NLRE_and_Regression_SDPThetaOnly(Model_SDP_NLRE_and_Regression):
    Name = 'Model_SDP_NLRE_and_Regression_SDPThetaOnly'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    --
    Designed to reconstruct only SDPTheta
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        super(Model_SDP_NLRE_and_Regression_SDPThetaOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        self.InWeights  = torch.tensor([1,0])
        self.OutWeights = torch.tensor([1,0])
        self.OutputMode = 'Both'
        # Apply the pre-trained weights if provided
        if 'RegressionBlockWeighs' in kwargs:
            pretrained_weights = kwargs['RegressionBlockWeighs']
            # print(pretrained_weights)
            self.load_state_dict(pretrained_weights)
            # self.Freeze_Regression_Block()


class Model_SDP_NLRE_and_Regression_SDPPhiOnly(Model_SDP_NLRE_and_Regression):
    Name = 'Model_SDP_NLRE_and_Regression_SDPPhiOnly'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    --
    Designed to reconstruct only SDPPhi
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        super(Model_SDP_NLRE_and_Regression_SDPPhiOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        self.InWeights  = torch.tensor([0,1])
        self.OutWeights = torch.tensor([0,1])
        self.OutputMode = 'Both'
        # Apply the pre-trained weights if provided
        if 'RegressionBlockWeighs' in kwargs:
            pretrained_weights = kwargs['RegressionBlockWeighs']
            self.load_state_dict(pretrained_weights)
            # self.Freeze_Regression_Block()



class Model_SDP_NLRE_and_Regression_RawLogits(Model_SDP_NLRE_and_Regression):
    Name = 'Model_SDP_NLRE_and_Regression_RawLogits'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    --
    Outputs raw logits for the classification with competition, not individual sigmoid activations
    '''

    # Init, same as parent
    
    def forward(self,Main,Aux,Augmentation_Scale = None, Augmentation_Function = 'UniformNearSample',Augmentation_Magnitude = None):
        self.device = self.Dense1.weight.device

        all_dict = {
            'Main': Main,
            'Aux': Aux,
            'Augmentation_Scale': Augmentation_Scale,
            'Augmentation_Function': Augmentation_Function,
            'Augmentation_Magnitude': Augmentation_Magnitude,
            'device': self.device
        }


        all_dict = self.parse_inputs(all_dict)
        all_dict = self.Regression_block(all_dict)

        if self.OutputMode == 'Regression':
            return [None,None,all_dict['Reg_RecVals']]
        
        all_dict = self.Classification_block(all_dict)
        if self.OutputMode == 'Classification': return [all_dict['Preds'],all_dict['Augmented_RecVals'],None]
        if self.OutputMode == 'Both': return [all_dict['Preds'],all_dict['Augmented_RecVals'],all_dict['Reg_RecVals']]
        raise ValueError(f'Output Mode {self.OutputMode} not recognized')

    def parse_inputs(self, all_dict):
        Main = all_dict['Main']
        device = all_dict['device']
        
        assert self.OutputMode in ['Both','Regression','Classification'], f'Output Mode {self.OutputMode} not recognized'
        if self.OutputMode != 'Regression':
            assert self.InWeights.sum() == 1, 'This Network is designed to reconstruct only one of Theta or Phi, Not Both. Set InWeights accordingly'
            assert self.OutWeights.sum() == 1, 'This Network is designed to reconstruct only one of Theta or Phi, Not Both. Set OutWeights accordingly'

        assert len(Main) == 2, 'Two Mains are expected'

        
        if Main[0].ndim == 4: # Main should have 3 dims, RecVals 2 dims , forwards order
            RecVals = Main[1].to(device)
            Main    = Main[0].to(device)
        elif Main[1].ndim == 4: # Backwards order
            RecVals = Main[0].to(device)
            Main    = Main[1].to(device)
        else:
            raise ValueError('Cannot determine Main and RecVals')
        all_dict['Main']    = Main
        all_dict['RecVals'] = RecVals
        return all_dict
    
    def Regression_block(self, all_dict):
        Main    = all_dict['Main']
        RecVals = all_dict['RecVals']
        device  = all_dict['device']

        Main_L = self.Conv_Activation(self.conv_0_large(Main))
        Main_S = self.Conv_Activation(self.conv_0_small(Main))
        Main = torch.cat([Main_L,Main_S],dim=1)
        Main = self.Conv_Dropout(self.Conv_Activation(self.conv_0(Main)))
        Main = self.BN_0(Main)
        
    
        Main = self.BN_1(self.Conv_Dropout(self.conv_1(Main)))
        Main = self.BN_2(self.Conv_Dropout(self.conv_2(Main)))
        Main = self.BN_3(self.Conv_Dropout(self.conv_3(Main)))
        Main = self.BN_4(self.Conv_Dropout(self.conv_4(Main)))
        Main = self.BN_5(self.Conv_Dropout(self.conv_5(Main)))

    
        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Activation(self.Dense3(Main))
       
        # Regression is done without augmentation
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg1(Main)))
        Theta_Reg = self.Dense_Dropout(self.Dense_Activation(self.SDPTheta_Reg2(Theta_Reg)))
        Theta_Reg = self.Angle_Activation(self.SDPTheta_Reg3(Theta_Reg))
        # Theta_Reg = self.SDPTheta_Reg3(Theta_Reg)

        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg1(Main)))
        Phi_Reg   = self.Dense_Dropout(self.Dense_Activation(self.SDPPhi_Reg2(Phi_Reg)))
        Phi_Reg   = self.Angle_Activation(self.SDPPhi_Reg3(Phi_Reg)) 
        # Phi_Reg   = self.SDPPhi_Reg3(Phi_Reg)

        Reg_RecVals = torch.cat([Theta_Reg * self.OutWeights[0].to(device),
                                 Phi_Reg   * self.OutWeights[1].to(device)
                                 ],dim=1) 
        
        all_dict['Main']    = Main
        all_dict['Theta_Reg'] = Theta_Reg
        all_dict['Phi_Reg']   = Phi_Reg
        all_dict['Reg_RecVals'] = Reg_RecVals
        
        return all_dict

    def Classification_block(self, all_dict):
        Main    = all_dict['Main']
        Theta_Reg = all_dict['Theta_Reg']
        Phi_Reg   = all_dict['Phi_Reg']
        # Reg_RecVals = all_dict['Reg_RecVals']
        device  = all_dict['device']
        
        if all_dict['Augmentation_Scale'] is None:
            all_dict['Augmentation_Scale'] = 30
            
        if (all_dict['Augmentation_Magnitude'] is None) and (all_dict['Augmentation_Function'] == 'UniformNearSample'):
            all_dict['Augmentation_Magnitude'] =  0.5*torch.pi/180.0 # 0.5 degrees in radians

        if all_dict['Augmentation_Function'] != 'NoAugmentation':
            if all_dict['Augmentation_Function'] == 'UniformNearSample':

                Preds_list             = []
                Augmented_RecVals_list = []

                Regression_Expectation_Theta = Theta_Reg.detach()
                Regression_Expectation_Theta = torch.atan2(Theta_Reg[:,0],Theta_Reg[:,1]).unsqueeze(1)
                Regression_Expectation_Phi   = Phi_Reg  .detach()
                Regression_Expectation_Phi   = torch.atan2(Phi_Reg  [:,0],Phi_Reg  [:,1]).unsqueeze(1)

                for aug_step in range(-all_dict['Augmentation_Scale']//2, all_dict['Augmentation_Scale']//2):
                    Aug_Rec_Theta = Regression_Expectation_Theta + aug_step * all_dict['Augmentation_Magnitude'] * self.InWeights[0]
                    Aug_Rec_Phi   = Regression_Expectation_Phi   + aug_step * all_dict['Augmentation_Magnitude'] * self.InWeights[1]
                
                    These_Aug_RecVals = torch.cat([Aug_Rec_Theta,Aug_Rec_Phi],dim=1)
                    Augmented_RecVals_list.append(These_Aug_RecVals)
                
                    # Aug Rec Vals Are the :
                    # - Augmented Reconstruction Value Theta/Phi, depending on the model we are Training
                    # - And the Regression Prediction for the Other value
                    
                    if self.OutWeights[0] == 1:
                        Aug_Rec_Theta = torch.cat([torch.sin(Aug_Rec_Theta),torch.cos(Aug_Rec_Theta)],dim=1)
                        Theta_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Aug_Rec_Theta],dim=1)
                        
                        Theta = self.Dense_Activation(self.SDPTheta_Cls1(Theta_Cls_Inp))
                        Theta = self.Dense_Activation(self.SDPTheta_Cls2(Theta))
                        Theta = self.SDPTheta_Cls3(Theta)  # Raw Logits
                    else:
                        Theta = torch.tensor([[0.0]]).repeat(Main.shape[0],1).to(device)

                    if self.OutWeights[1] == 1:
                        Aug_Rec_Phi = torch.cat([torch.sin(Aug_Rec_Phi),torch.cos(Aug_Rec_Phi)],dim=1)
                        Phi_Cls_Inp = torch.cat([Main,Theta_Reg,Phi_Reg,Aug_Rec_Phi],dim=1)

                        Phi = self.Dense_Activation(self.SDPPhi_Cls1(Phi_Cls_Inp))
                        Phi = self.Dense_Activation(self.SDPPhi_Cls2(Phi))
                        Phi = self.SDPPhi_Cls3(Phi)  # Raw Logits
                    else:
                        Phi = torch.tensor([[0.0]]).repeat(Main.shape[0],1).to(device)
                    
                    These_Preds = torch.cat([Theta,Phi],dim=1)
                    Preds_list.append(These_Preds)
            else:
                raise NotImplementedError(f'Augmentation Function {all_dict["Augmentation_Function"]} not implemented')
            
            Preds             = torch.stack(Preds_list            ,dim=1) # Raw Logits
            Augmented_RecVals = torch.stack(Augmented_RecVals_list,dim=1) # Corresponding Rec Vals

            Preds = torch.softmax(Preds,dim=1)  # Convert to Probabilities
            Preds             = Preds            .permute(0,1,2).reshape(-1,self.in_RecValues_channels)
            Augmented_RecVals = Augmented_RecVals.permute(0,1,2).reshape(-1,self.in_RecValues_channels)


            # This reshaping works with the current loss functions.
            all_dict['Preds']             = Preds
            all_dict['Augmented_RecVals'] = Augmented_RecVals
            return all_dict
        else:
            raise NotImplementedError('NoAugmentation not implemented for Raw Logits Model')


class Model_SDP_NLRE_and_Regression_RawLogits_SDPThetaOnly(Model_SDP_NLRE_and_Regression_RawLogits):
    Name = 'Model_SDP_NLRE_and_Regression_RawLogits_SDPThetaOnly'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    --
    Designed to reconstruct only SDPTheta
    Outputs raw logits for the classification with competition, not individual sigmoid activations
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        super(Model_SDP_NLRE_and_Regression_RawLogits_SDPThetaOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        self.InWeights  = torch.tensor([1,0])
        self.OutWeights = torch.tensor([1,0])
        self.OutputMode = 'Both'
        # Apply the pre-trained weights if provided
        if 'RegressionBlockWeighs' in kwargs:
            pretrained_weights = kwargs['RegressionBlockWeighs']
            self.load_state_dict(pretrained_weights)
            # self.Freeze_Regression_Block()

class Model_SDP_NLRE_and_Regression_RawLogits_SDPPhiOnly(Model_SDP_NLRE_and_Regression_RawLogits):
    Name = 'Model_SDP_NLRE_and_Regression_RawLogits_SDPPhiOnly'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv2d Layers in blocks with residual connections
    Reconstruction is done for single telescope
    Outputs continuous values for Theta and Phi - Used in training. Regression Loss should be used here
    Also Outputs the likelihood of the reconstruction being correct - Used in inference. Binary Cross Entropy Loss should be used here
    --
    Designed to reconstruct only SDPPhi
    Outputs raw logits for the classification with competition, not individual sigmoid activations
    '''

    def __init__(self,in_main_channels = (2,2), N_kernels = 64, N_dense_nodes = 256, **kwargs):
        super(Model_SDP_NLRE_and_Regression_RawLogits_SDPPhiOnly, self).__init__(in_main_channels, N_kernels, N_dense_nodes, **kwargs)
        self.InWeights  = torch.tensor([0,1])
        self.OutWeights = torch.tensor([0,1])
        self.OutputMode = 'Both'
        # Apply the pre-trained weights if provided
        if 'RegressionBlockWeighs' in kwargs:
            pretrained_weights = kwargs['RegressionBlockWeighs']
            self.load_state_dict(pretrained_weights)
            # self.Freeze_Regression_Block()