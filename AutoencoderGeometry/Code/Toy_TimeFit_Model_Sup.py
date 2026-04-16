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

def TimeFitEq_Toy(chi_i,chi_0,Rp,T0,FitType = 'Tan'):
    if FitType == 'Tan':
        if type(chi_i) == torch.Tensor:
            assert type(chi_0) == torch.Tensor and type(Rp) == torch.Tensor and type(T0) == torch.Tensor, "All inputs must be of the same type"
            return T0+Rp*torch.tan((chi_0-chi_i)/2)
        elif type(chi_i) == np.ndarray:
            return T0+Rp*np.tan((chi_0-chi_i)/2)
        else:
            raise TypeError("Input must be either a torch.Tensor or a numpy.ndarray")

    if FitType == 'Linear':
        return T0+Rp*(chi_0-chi_i)
        



def Loss(Pred,Truth,keys = ['Chi_0','Rp','T0'],ReturnTensor = True,Debug_Mode = False, **kwargs):
    ''''
    Will calculate the loss for the time profile and geometry predictions
    '''

    if type(Pred) == dict:
        Time_Profile_Pred = Pred['time_profile']
        Geometry_Pred     = Pred['geometry']
        Test_Chi_is       = Pred['test_chi_is'].squeeze(-1) # (B, 10)
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

            # print(f'Chi_0s shape', Chi_0s.shape)
            # print(f'Rps shape', Rps.shape)
            # print(f'T0s shape', T0s.shape)
            # print(f'Test_Chi_is shape', Test_Chi_is.shape)
            # print(F'Time_Profile_Pred shape', Time_Profile_Pred.shape)

            Time_Profile_Truth = TimeFitEq_Toy(Test_Chi_is, Chi_0s, Rps, T0s) # (B, 10)

            losses['Profile_chii'] = F.mse_loss(Time_Profile_Pred[:,:,0], Test_Chi_is)
            losses['Profile_time'] = F.mse_loss(Time_Profile_Pred[:,:,1], Time_Profile_Truth)
            
        else:
            losses['Profile_chii'] = torch.zeros(1).to(device)
            losses['Profile_time'] = torch.zeros(1).to(device)

        if Train_Type in ['Geometry','Both']:
            for i,key in enumerate(keys):
                losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i].to(device))
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

    Profile_preds  = []
    Test_Chi_is    = []
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
                Test_Chi_is  .append(Model_out['test_chi_is' ].to('cpu'))

            if Train_Type in ['Geometry','Both']:
                Geometry_preds.append(Model_out['geometry'].to('cpu'))
            
            Truths.append(BatchTruth.to('cpu'))
            if Debug_Mode: break
        if Train_Type in ['Profile','Both']:
            Profile_preds = torch.cat(Profile_preds,dim=0)
            Test_Chi_is   = torch.cat(Test_Chi_is  ,dim=0)
        else:
            Profile_preds = None
        if Train_Type in ['Geometry','Both']:
            Geometry_preds = torch.cat(Geometry_preds,dim=0)
        else:
            Geometry_preds = None

        Truths = torch.cat(Truths,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    return Loss({'time_profile': Profile_preds, 'geometry': Geometry_preds, 'test_chi_is':Test_Chi_is}, Truths, keys=Dataset.Truth_Keys, ReturnTensor=False)

def metric(model,Dataset,device,keys = ['Chi_0','Rp','T0'],BatchSize = 512,Debug_Mode = False, **kwargs):
    ''' For now i am not sure what to do'''
    if Debug_Mode: print('Calculating Metrics..., no debug mode -  running normally')
    metrics = {}
    metrics['Units'] = ['s','rad','m','ns']

    for key in keys:
        metrics[key] = 0
    metrics['Profile'] = 0

    return metrics


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


class Model_Toy_Autoencoder_TimeFit(nn.Module):
    Name = 'Model_Toy_Autoencoder_TimeFit'
    Description = '''Model will try to figure out a state representation of the shower time profile
    Reconstruction of geometry is done from this state
    Training to be done in two steps
    1. train the autoencoder to reconstruct the time profile
        -  to do this, model will observe the data and try to produce 10 new values from the profile
    2. train the decoder that will use the latent state to reconstruct the geometry
        
    '''
    

    def __init__(self, in_main_channels =(2,), pixel_embedding_size = 32, latent_space_size = 32,N_dense_nodes = 64,Train_Type = 'Both',**kwargs):
        super(Model_Toy_Autoencoder_TimeFit, self).__init__()
        
        self.Pix_Features = in_main_channels[0]
        self.kwargs = kwargs
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)
        self.max_latent_iterations = kwargs.get('max_latent_iterations', 10)
        self.latent_space_change_scale = kwargs.get('latent_space_change_scale', 0.01)
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
        self.recursive_latent_space_update = nn.Linear(latent_space_size+pixel_embedding_size, latent_space_size)

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
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_i'],Graph))),device=device).int()
        # I think this produces the exact thing i need
        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        
        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_i'], event['time']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chi_i'], event['station_time']], device=device) 

        pix_embedding =  self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
        latent_space  = self.attention_pool(pix_embedding, Mask) # (B, latent_space_size)
        latent_space  = self.construct_latent_space(latent_space) # (B, latent_space_size)

        old_latent_space = torch.zeros_like(latent_space)
        N_iter = 0
        while (latent_space - old_latent_space).sum() > latent_space.sum()*self.latent_space_change_scale and N_iter < self.max_latent_iterations:
            old_latent_space = latent_space
            # For each pixel, concatenate the latent space to the pixel embedding and process it
            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, pix_embedding.shape[1], -1)
            recursive_input = torch.cat([pix_embedding, latent_space_expanded], dim=-1) # (B, N_pix, pixel_embedding_size + latent_space_size)
            recursive_output = self.recursive_latent_space_update(recursive_input) # (B, N_pix, latent_space_size)
            # Now pool this again to get the new latent space
            latent_space = self.attention_pool(recursive_output, Mask) # (B, latent_space_size)
            
            N_iter += 1


        # Loss should take care of these debug prints
        if self.Train_Type in ['Profile','Both']:
            # Now make the preditions for the time profile
            # for each event generate 10 u coordinates between 0 and chi_0
            
            # Chi_0s = Aux[:,0].unsqueeze(1).to(device) # (B, 1) # Kinda cheating but its not a problem, cause i can resample the pixel space values instead and it will be good
            
            # test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
            # test_chis = (test_chis * Chi_0s).unsqueeze(-1) # (B, 10) # between 0 and chi_0
            
            max_chis = torch.tensor([event['chi_i'].max() for event in Graph], device=device).unsqueeze(1) # (B, 1)
            # More correct way of doing this where test values are between 0 and pi
            test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
            test_chis = (test_chis * max_chis).unsqueeze(-1) # (B, 10) # between 0 and chi_0

            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)

            time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
            B,S,D = time_profile_input.shape
            
            time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
            time_profile = time_profile.reshape(B,S,2)
            
        else:
            time_profile = None
            test_chis    = None

        if self.Train_Type in ['Geometry','Both']:
            geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights.to(device) # (B, 3)
        else:
            geometry = None

        

        return {
            'time_profile': time_profile, # (B, 10, 2)
            'test_chi_is' : test_chis,    # (B, 10, 1)
            'geometry'    : geometry    , # (B, 2)
            'N_iter'      : N_iter
        }
    
        # I can regenerate the expected chi_is in loss, cause i know the chi_0s
        # For now this is just for continuity, i will start with only training the latent space construction 
