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
        Truth_Classes = (torch.abs(RecValues - Truth_RecValues) < 2.1*torch.pi/180).float() # within 1/3.5 degrees which is slightly larger than the augmentation step of 1/4 degree
        
        # Now, calculate the weights for each guess
        Augmentation_magnitude = torch.abs(RecValues - Truth_RecValues)
        
        weights = torch.ones_like(Truth_Classes)
        label_T = Truth_Classes == 1
        label_F = Truth_Classes == 0 
        weights[label_T] = Augmentation_Scale/torch.sum(label_T).float() # Will give equal KINDA weight to pos/neg samples
        weights[label_F] = Augmentation_magnitude[label_F]* 5 * Pred_Classes[label_F]
        weights = weights.detach()
        

        
        
        
    
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
            Model_out = model(BatchMains,BatchAux,Augmentation_Scale=2)
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


def time_fit(chii,chi_0,Rp,T0):
    in_tan  = (chi_0 - chii)/2
    scale = Rp/(3e8)
    time_profile = T0/1e9 + scale*torch.tan(in_tan)
    return time_profile*1e7 # return in 100ns units - telescope bins
    

def Loss(Pred,Truth,keys = ['Chi_0','Rp','T0'],ReturnTensor = True,Debug_Mode = False, **kwargs):
    ''''
    Will calculate the loss for the time profile and geometry predictions
    '''

    if type(Pred) == dict:
        Time_Profile_Pred = Pred['time_profile']
        Geometry_Pred     = Pred['geometry']
    else:
        raise ValueError('Predictions should be a dictionary with keys time_profile and geometry')

    Train_Type = 'Both'
    if Time_Profile_Pred is None: Train_Type = 'Geometry'
    if Geometry_Pred     is None: Train_Type = 'Profile'

    if not Debug_Mode:
        losses = {}
        
        device = Time_Profile_Pred.device if Time_Profile_Pred is not None else Geometry_Pred.device

        if Train_Type in ['Profile','Both']:
            
            Chi_0s = Truth[:,0].unsqueeze(1).to(device)
            Rps    = Truth[:,1].unsqueeze(1).to(device)
            T0s    = Truth[:,2].unsqueeze(1).to(device)

            test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(len(Time_Profile_Pred), 1) # (B, 10) # between 0 and 1
            test_chis = (test_chis * Chi_0s) # (B, 10) # between 0 and chi_0
            
            Time_Profile_Truth = time_fit(test_chis, Chi_0s, Rps, T0s) # (B, 10)

            losses['Profile_chii'] = F.mse_loss(Time_Profile_Pred[:,:,0], test_chis)
            losses['Profile_time'] = F.mse_loss(Time_Profile_Pred[:,:,1], Time_Profile_Truth)
            
        else:
            losses['Profile_chii'] = torch.zeros(1).to(device)
            losses['Profile_time'] = torch.zeros(1).to(device)

        if Train_Type in ['Geometry','Both']:
            for i,key in enumerate(keys):
                losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i])
        else:
            for i,key in enumerate(keys):
                losses[key] = torch.zeros(1).to(device)

        losses['Total'] = sum(losses.values())

        if ReturnTensor: return losses
        else:
            losses = {key:loss.item() for key,loss in losses.items()}
            return losses
        

    elif Debug_Mode: 
        print('################ CALCULATING LOSS ################')
        print(f'    Found Train Type: {Train_Type}')
        losses = {}
        
        device = Time_Profile_Pred.device if Time_Profile_Pred is not None else Geometry_Pred.device

        if Train_Type in ['Profile','Both']:
            
            Chi_0s = Truth[:,0].unsqueeze(1).to(device)
            Rps    = Truth[:,1].unsqueeze(1).to(device)
            T0s    = Truth[:,2].unsqueeze(1).to(device)

            test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(len(Time_Profile_Pred), 1) # (B, 10) # between 0 and 1
            test_chis = (test_chis * Chi_0s) # (B, 10) # between 0 and chi_0
                
            print(f' shape of all values in the time fit calculation')
            Time_Profile_Truth = time_fit(test_chis, Chi_0s, Rps, T0s) # (B, 10)
            losses['Profile_chii'] = F.mse_loss(Time_Profile_Pred[:,:,0], test_chis)
            losses['Profile_time'] = F.mse_loss(Time_Profile_Pred[:,:,1], Time_Profile_Truth)

            
        
            print(f' Profile Calculation, Found the following inputs in event 0')
            print(f'    Chi_0s: {Chi_0s[0]}')
            print(f'    Rps: {Rps[0]}')
            print(f'    T0s: {T0s[0]}')

            print(f'    test_chis               : {test_chis[0]}')
            print(f'    Time_Profile_Pred[0,:,0]: {Time_Profile_Pred[0,:,0]}')
            print(f'    Time_Profile_Truth      : {Time_Profile_Truth[0]}')
            print(f'    Time_Profile_Pred[0,:,1]: {Time_Profile_Pred[0,:,1]}')
            print(f'    Chii Profile MSE Loss: {losses["Profile_chii"].item()}')
            print(f'    Time Profile MSE Loss: {losses["Profile_time"].item()}')



        else:
            losses['Profile_chii'] = torch.zeros(1).to(device)
            losses['Profile_time'] = torch.zeros(1).to(device)

        if Train_Type in ['Geometry','Both']:
            for i,key in enumerate(keys):
                losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i])
                print(f' Geometry Calculation for {key}, Found the following inputs')
                print(f'    Geometry_Pred[:,{i}]: {Geometry_Pred[:,i]}')
                print(f'    Truth[:,{i}]: {Truth[:,i]}')
                print(f'    MSE Loss for {key}: {losses[key].item()}')


        else:
            for i,key in enumerate(keys):
                losses[key] = torch.zeros(1).to(device)

        losses['Total'] = sum(losses.values())

        if ReturnTensor: return losses
        else:
            losses = {key:loss.item() for key,loss in losses.items()}
            return losses


def validate(model,Dataset,Loss,device,BatchSize = 512, Debug_Mode = False, **kwargs):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    if Debug_Mode: print('################ VALIDATION ################')

    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    Dataset.BatchSize = 1 if Debug_Mode else BatchSize
    
    Train_Type = None

    Profile_preds   =[]
    Geometry_preds = []
    Truths         = []
    
    with torch.no_grad():
        for i,(_, BatchMains, BatchAux, BatchTruth, _)  in enumerate(Dataset):
            Model_out = model(BatchMains,BatchAux)
            if i == 0:
                if   Model_out['time_profile'] is None: Train_Type = 'Geometry'
                elif Model_out['geometry']     is None: Train_Type = 'Profile'
                else:                                   Train_Type = 'Both'

            if Train_Type in ['Profile','Both']:
                Profile_preds.append(Model_out['time_profile'].to('cpu'))
            
            if Train_Type in ['Geometry','Both']:
                Geometry_preds.append(Model_out['geometry'].to('cpu'))
            
            Truths.append(BatchTruth.to('cpu'))
            if Debug_Mode: break
        if Train_Type in ['Profile','Both']:
            Profile_preds = torch.cat(Profile_preds,dim=0)
        else:
            Profile_preds = None
        if Train_Type in ['Geometry','Both']:
            Geometry_preds = torch.cat(Geometry_preds,dim=0)
        else:
            Geometry_preds = None

        Truths = torch.cat(Truths,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss({'time_profile': Profile_preds, 'geometry': Geometry_preds}, Truths, keys=Dataset.Truth_Keys, ReturnTensor=False)


def metric(model,Dataset,device,keys = ['Chi_0','Rp','T0'],BatchSize = 512,Debug_Mode = False, **kwargs):
    ''' For now i am not sure what to do'''
    if Debug_Mode: print('Calculating Metrics..., no debug mode -  running normally')
    metrics = {}
    metrics['Units'] = ['s','rad','m','ns']

    for key in keys:
        metrics[key] = 0
    metrics['Profile'] = 0

    return metrics
    
# -------------------------
# Attention pooling
# -------------------------

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, mask):
        # h   : (Batch, N_pix, latent_space_Dim) N_pix = hits + padding
        # mask: (Batch, N_pix)  (1 = valid hit, 0 = padding)

        scores = self.score(h).squeeze(-1)          # (B, N)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=1)            # (B, N)

        z = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, D)
        return z


class Model_Autoencoder_TimeFit(nn.Module):
    Name = 'Model_Autoencoder_TimeFit'
    Description = '''Model will try to figure out a state representation of the shower time profile
    Reconstruction of geometry is done from this state
    Training to be done in two steps
    1. train the autoencoder to reconstruct the time profile
        -  to do this, model will observe the data and try to produce 10 new values from the profile
    2. train the decoder that will use the latent state to reconstruct the geometry
        
    '''
    

    def __init__(self, in_main_channels =(3,), pixel_embedding_size = 32, latent_space_size = 32,N_dense_nodes = 64,Train_Type = 'Profile',**kwargs):
        super(Model_Autoencoder_TimeFit, self).__init__()
        
        self.Pix_Features = in_main_channels[0]
        self.kwargs = kwargs
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)

        if self.Debug_Mode: 
            print(f'    Model initialized with Train Type: {self.Train_Type} and Debug Mode: {self.Debug_Mode}')
            print(f'    Trimming down the model sizes')
            pixel_embedding_size = 8
            latent_space_size    = 8
            N_dense_nodes        = 16
        # Encode in the Pixel's dimension - size-out (B, N_pix, latent_space_size)
        self.pix_process = nn.Sequential(
            nn.Linear(self.Pix_Features, pixel_embedding_size),
            nn.LeakyReLU(),
            nn.Linear(pixel_embedding_size, pixel_embedding_size),
            nn.LeakyReLU(),
        )
        

        self.attention_pool = AttentionPooling(pixel_embedding_size)

        self.construct_latent_space = nn.Linear(pixel_embedding_size, latent_space_size)

        self.produce_time_profile = nn.Sequential(
            nn.Linear(latent_space_size + 1, N_dense_nodes), # +1 for the u coordinate
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 2) # predict t and q
        )

        self.produce_geometry = nn.Sequential(
            nn.Linear(latent_space_size, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 3) # predict Chi_0* and log_Rp and T0
        )

        self.Geometry_OutWeights = torch.tensor([1,1,1])
            
    def forward(self,Graph,Aux):
        if self.Debug_Mode and self.training:
            return self.debug_forward(Graph,Aux)
        else:
            return self.clean_forward(Graph,Aux)
        
    def clean_forward(self,Graph,Aux):
        device = self.pix_process[0].weight.device

        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        # I think this produces the exact thing i need
        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        
        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_is'], event['time'], event['charge']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chii'], event['station_time'], event['station_signal']], device=device) 

        pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
        latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
        latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)

        # Loss should take care of these debug prints
        if self.Train_Type in ['Profile','Both']:
            # Now make the preditions for the time profile
            # for each event generate 10 u coordinates between 0 and chi_0
            
            Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
            
            test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
            test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
            
            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)

            time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
            B,S,D = time_profile_input.shape
            
            time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
            time_profile = time_profile.reshape(B,S,2)
            
        else:
            time_profile = None

        if self.Train_Type in ['Geometry','Both']:
            geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights # (B, 3)
        else:
            geometry = None

        

        return {
            'time_profile': time_profile, # (B, 10, 2)
            'geometry'    : geometry    , # (B, 2)
        }
    
        # I can regenerate the expected chi_is in loss, cause i know the chi_0s
        # For now this is just for continuity, i will start with only training the latent space construction 

    def debug_forward(self,Graph,Aux):

        device = self.pix_process[0].weight.device

        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        # I think this produces the exact thing i need
        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        if self.Debug_Mode and self.training:
            print(f'    Forward Pass, Found the following inputs')
            print(f'    N_Events: {N_Events}')
            print(f'    N_pix_in_event: {N_pix_in_event}')        
            print(f'    Mask: {Mask}')        

        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_is'], event['time'], event['charge']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chii'], event['station_time'], event['station_signal']], device=device) 

        
        if self.Debug_Mode and self.training:
            print(f'    Pix_Data: {Pix_Data}')
            pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
            print(f'    pix_embedding: {pix_embedding}')
            latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
            print(f'    latent_space after attention pool: {latent_space}')
            latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)
            print(f'    latent_space after construction: {latent_space}')

        else:
            pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
            latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
            latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)

        # Loss should take care of these debug prints
        if self.Train_Type in ['Profile','Both']:
            # Now make the preditions for the time profile
            # for each event generate 10 u coordinates between 0 and chi_0
            if self.Debug_Mode and self.training:
                print(f'    Generating Time Profile Predictions, Found the following inputs')
                Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
                print(f'    Chi_0s: {Chi_0s}')
                test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
                print(f'    test_chis before scaling: {test_chis}')
                test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
                print(f'    test_chis after scaling: {test_chis}')
                
                latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)
                print(f'    latent_space_expanded: {latent_space_expanded}')

                time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
                print(f'    time_profile_input: {time_profile_input}')
                B,S,D = time_profile_input.shape
                
                time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
                print(f'    time_profile before reshape: {time_profile}')
                time_profile = time_profile.reshape(B,S,2)
                print(f'    time_profile after reshape: {time_profile}')
            else:
                Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
                
                test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
                test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
                
                latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)

                time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
                B,S,D = time_profile_input.shape
                
                time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
                time_profile = time_profile.reshape(B,S,2)
            
        else:
            time_profile = None

        if self.Train_Type in ['Geometry','Both']:
            geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights # (B, 3)
        else:
            geometry = None

        

        return {
            'time_profile': time_profile, # (B, 10, 2)
            'geometry'    : geometry    , # (B, 2)
        }
    
        # I can regenerate the expected chi_is in loss, cause i know the chi_0s
        # For now this is just for continuity, i will start with only training the latent space construction 




class Model_Autoencoder_TimeFit_withGeometry(nn.Module):
    Name = 'Model_Autoencoder_TimeFit_withGeometry'
    Description = '''Model will try to figure out a state representation of the shower time profile
    Reconstruction of geometry is done from this state
    Training to be done in two steps
    1. train the autoencoder to reconstruct the time profile
        -  to do this, model will observe the data and try to produce 10 new values from the profile
    2. train the decoder that will use the latent state to reconstruct the geometry
        
    '''
    

    def __init__(self, in_main_channels =(3,), pixel_embedding_size = 32, latent_space_size = 32,N_dense_nodes = 64,Train_Type = 'Profile',**kwargs):
        super(Model_Autoencoder_TimeFit_withGeometry, self).__init__()
        
        self.Pix_Features = in_main_channels[0]
        self.kwargs = kwargs
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)

        if self.Debug_Mode: 
            print(f'    Model initialized with Train Type: {self.Train_Type} and Debug Mode: {self.Debug_Mode}')
            print(f'    Trimming down the model sizes')
            pixel_embedding_size = 8
            latent_space_size    = 8
            N_dense_nodes        = 16
        # Encode in the Pixel's dimension - size-out (B, N_pix, latent_space_size)
        self.pix_process = nn.Sequential(
            nn.Linear(self.Pix_Features, pixel_embedding_size),
            nn.LeakyReLU(),
            nn.Linear(pixel_embedding_size, pixel_embedding_size),
            nn.LeakyReLU(),
        )
        

        self.attention_pool = AttentionPooling(pixel_embedding_size)

        self.construct_latent_space = nn.Linear(pixel_embedding_size, latent_space_size)

        self.produce_time_profile = nn.Sequential(
            nn.Linear(latent_space_size + 1+3, N_dense_nodes), # +1 for the u coordinate +3 for the true geometry values
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 2) # predict t and q
        )

        self.produce_geometry = nn.Sequential(
            nn.Linear(latent_space_size, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 3) # predict Chi_0* and log_Rp and T0
        )

        self.Geometry_OutWeights = torch.tensor([1,1,1])
            
    def forward(self,Graph,Aux):
        if self.Debug_Mode and self.training:
            return self.debug_forward(Graph,Aux)
        else:
            return self.clean_forward(Graph,Aux)
        
    def clean_forward(self,Graph,Aux):
        device = self.pix_process[0].weight.device

        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        # I think this produces the exact thing i need
        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        
        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_is'], event['time'], event['charge']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chii'], event['station_time'], event['station_signal']], device=device) 

        pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
        latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
        latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)

        # Loss should take care of these debug prints
        if self.Train_Type in ['Profile','Both']:
            # Now make the preditions for the time profile
            # for each event generate 10 u coordinates between 0 and chi_0
            Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
            Rps    = Aux[:,5].unsqueeze(1).to(device)
            T0s    = Aux[:,6].unsqueeze(1).to(device)

            test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
            test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
            
            #Now we add true geometry values to the latent space
            True_Geometry = torch.cat([Chi_0s, Rps, T0s], dim=-1) # (B, 3)
            latent_space = torch.cat([latent_space, True_Geometry], dim=-1) # (B, latent_space_size + 3)
            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)
            
            time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
            B,S,D = time_profile_input.shape
            
            time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
            time_profile = time_profile.reshape(B,S,2)
            
        else:
            time_profile = None

        if self.Train_Type in ['Geometry','Both']:
            geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights # (B, 3)
        else:
            geometry = None

        

        return {
            'time_profile': time_profile, # (B, 10, 2)
            'geometry'    : geometry    , # (B, 2)
        }
    
        # I can regenerate the expected chi_is in loss, cause i know the chi_0s
        # For now this is just for continuity, i will start with only training the latent space construction 

    def debug_forward(self,Graph,Aux):
        raise NotImplementedError('Debug forward not implemented for this model yet, cause its a bit more work to print the geometry inputs and outputs, will do if needed')
        device = self.pix_process[0].weight.device

        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        # I think this produces the exact thing i need
        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        if self.Debug_Mode and self.training:
            print(f'    Forward Pass, Found the following inputs')
            print(f'    N_Events: {N_Events}')
            print(f'    N_pix_in_event: {N_pix_in_event}')        
            print(f'    Mask: {Mask}')        

        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_is'], event['time'], event['charge']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chii'], event['station_time'], event['station_signal']], device=device) 

        
        if self.Debug_Mode and self.training:
            print(f'    Pix_Data: {Pix_Data}')
            pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
            print(f'    pix_embedding: {pix_embedding}')
            latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
            print(f'    latent_space after attention pool: {latent_space}')
            latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)
            print(f'    latent_space after construction: {latent_space}')

        else:
            pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
            latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
            latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)

        # Loss should take care of these debug prints
        if self.Train_Type in ['Profile','Both']:
            # Now make the preditions for the time profile
            # for each event generate 10 u coordinates between 0 and chi_0
            if self.Debug_Mode and self.training:
                print(f'    Generating Time Profile Predictions, Found the following inputs')
                Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
                print(f'    Chi_0s: {Chi_0s}')
                test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
                print(f'    test_chis before scaling: {test_chis}')
                test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
                print(f'    test_chis after scaling: {test_chis}')
                
                latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)
                print(f'    latent_space_expanded: {latent_space_expanded}')

                time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
                print(f'    time_profile_input: {time_profile_input}')
                B,S,D = time_profile_input.shape
                
                time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
                print(f'    time_profile before reshape: {time_profile}')
                time_profile = time_profile.reshape(B,S,2)
                print(f'    time_profile after reshape: {time_profile}')
            else:
                Chi_0s = Aux[:,4].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
                
                test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
                test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
                
                latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)

                time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
                B,S,D = time_profile_input.shape
                
                time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
                time_profile = time_profile.reshape(B,S,2)
            
        else:
            time_profile = None

        if self.Train_Type in ['Geometry','Both']:
            geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights # (B, 3)
        else:
            geometry = None

        

        return {
            'time_profile': time_profile, # (B, 10, 2)
            'geometry'    : geometry    , # (B, 2)
        }
    
        # I can regenerate the expected chi_is in loss, cause i know the chi_0s
        # For now this is just for continuity, i will start with only training the latent space construction 

