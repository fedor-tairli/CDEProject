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

def time_fit(chii,chi_0,Rp,T0):
    in_tan  = (chi_0 - chii)/2
    scale = Rp/(3e8)
    time_profile = T0/1e9 + scale*torch.tan(in_tan)
    return time_profile*1e7 # return in 100ns units - telescope bins

def merge_parameters(PriorityParameters, SecondaryParameters):
    return {**SecondaryParameters, **PriorityParameters} # This overwrites the Secondary if there is same Priority key


def Loss(Pred,Truth, Loss_info,  **kwargs):
    ''''
    Will calculate the loss for the time profile and geometry predictions
    '''
    Loss_info = merge_parameters(Loss_info, kwargs) 
    # ---- Parse Kwargs ----
    keys         = Loss_info.get('keys'         , ['Chi_0','Rp','T0'])
    ReturnTensor = Loss_info.get('ReturnTensor' , True  ) 
    Debug_Mode   = Loss_info.get('Debug_Mode'   , False )

    # ---- Parse the predictions ----
    if type(Pred) == dict:
        Time_Profile_Pred = Pred['time_profile']
        Geometry_Pred     = Pred['geometry']
        Test_Chi_is       = Pred['test_chi_is'].squeeze(-1) # (B, 10)
        Acc_Loss          = Pred.get('Acc_Loss', torch.tensor(0.0)) # This is the accumulated loss of the time profile reconstruction during the recursive latent space update, it can be used for analysis and maybe for making decisions on iteration exit in future versions
        Time_Norm         = Pred.get('Time_Norm', torch.zeros(len(Truth))) # This is the normalization factor for the time values
    else:
        raise ValueError('Predictions should be a dictionary with keys time_profile and geometry')
    
    # ---- Loss Parameters ---- 
    T_G_Loss_ratio  = Loss_info.get('T_G_Loss_ratio' , torch.tensor(1.0  ) ) # geom vs profile loss ratio # For now i will disregard geometry in the final loss
    T_Loss_scale    = Loss_info.get('T_Loss_scale'   , torch.tensor(1e-4 ) ) # Normalisation item
    Rec_Loss_Weight = Loss_info.get('Rec_Loss_Weight', torch.tensor(3.0  ) ) # weight of this loss wrt the acc loss in iteration
    # print(f'    Loss Function - T_G_Loss_ratio: {T_G_Loss_ratio}, T_Loss_scale: {T_Loss_scale}, Rec_Loss_Weight: {Rec_Loss_Weight}')
    
    Train_Type = 'Both'
    if Time_Profile_Pred is None: Train_Type = 'Geometry'
    if Geometry_Pred     is None: Train_Type = 'Profile'

    
    losses = {}
    
    device = Time_Profile_Pred.device if Time_Profile_Pred is not None else Geometry_Pred.device

    if Train_Type in ['Profile','Both']:
        
        Chi_0s = Truth[:,0].unsqueeze(1).to(device)
        Rps    = Truth[:,1].unsqueeze(1).to(device)
        T0s    = Truth[:,2].unsqueeze(1).to(device)

        Time_Profile_Truth = time_fit(Test_Chi_is, Chi_0s, Rps, T0s) # (B, 10)
        Time_Profile_Truth = Time_Profile_Truth - Time_Norm.unsqueeze(1) # Min Time is subtracted from pixels - So do it here too

        # losses['Profile_chii'] = F.mse_loss(Time_Profile_Pred[:,:,0], Test_Chi_is)
        
        losses['Profile_time_Acc'] = Acc_Loss.to(device)
        losses['Profile_time_Rec'] = F.mse_loss(Time_Profile_Pred[:,:,0], Time_Profile_Truth)*Rec_Loss_Weight.to(device)
        losses['Profile_time']     = losses['Profile_time_Acc'] + losses['Profile_time_Rec'] 

    else:
        # losses['Profile_chii'] = torch.zeros(1).to(device)
        losses['Profile_time'] = torch.zeros(1).to(device)


    



    if Train_Type in ['Geometry','Both']:
        for i,key in enumerate(keys):
            if key == 'T0':
                losses[key] = F.mse_loss(Geometry_Pred[:,i]+ Time_Norm *1e9/1e7, Truth[:,i].to(device)) # Pixel Time Shift in units of ns 
            else:
                losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i].to(device))
    else:
        for i,key in enumerate(keys):
            losses[key] = torch.zeros(1).to(device)



    losses['Total'] = T_G_Loss_ratio*T_Loss_scale*losses['Profile_time'] + (torch.tensor(1.0)-T_G_Loss_ratio)*sum(losses[key] for key in keys)

    if ReturnTensor: return losses
    else:
        losses = {key:loss.item() for key,loss in losses.items()}
        return losses



def validate(model,Dataset,Loss, Loss_info, **kwargs):
    '''
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    Loss_info = merge_parameters(Loss_info, kwargs)
    Debug_Mode = Loss_info.get('Debug_Mode', False)

    if Debug_Mode: print('################ VALIDATION ################')

    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    TrainingBatchSize = Dataset.BatchSize
    # Dataset.BatchSize = 1 if Debug_Mode else BatchSize  # CARE THIS THING AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    
    Train_Type = None

    Profile_preds  = []
    Test_Chi_is    = []
    Geometry_preds = []
    Truths         = []
    Acc_Losses     = [] # Not interested in this for the validation, but doing anyway, just in case
    Time_Norms     = [] # very important 
    
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
                Acc_Losses   .append(Model_out.get('Acc_Loss', torch.zeros(BatchTruth.shape[0])).to('cpu'))

            if Train_Type in ['Geometry','Both']:
                Geometry_preds.append(Model_out['geometry'].to('cpu'))
            
            Time_Norms.append(Model_out.get('Time_Norm', torch.zeros(BatchTruth.shape[0])).to('cpu'))
            Truths.append(BatchTruth.to('cpu'))

            if Debug_Mode: break

        
        
        if Train_Type in ['Profile','Both']:
            Profile_preds = torch.cat(Profile_preds,dim=0)
            Test_Chi_is   = torch.cat(Test_Chi_is  ,dim=0)
            Acc_Losses    = torch.mean(torch.stack(Acc_Losses)) # Average Accumulated Loss across the validation set
        else:
            Profile_preds = None
        
        
        if Train_Type in ['Geometry','Both']:
            Geometry_preds = torch.cat(Geometry_preds,dim=0)
        else:
            Geometry_preds = None

        Time_Norms = torch.cat(Time_Norms,dim=0)
        Truths     = torch.cat(Truths    ,dim=0)

    # Return Batch Size to old value
    Dataset.BatchSize = TrainingBatchSize
    Loss_inp_dict = {
        'time_profile': Profile_preds,
        'geometry'    : Geometry_preds,
        'test_chi_is' : Test_Chi_is,
        'Acc_Loss'    : Acc_Losses,
        'Time_Norms'  : Time_Norms
    }
    return Loss(Loss_inp_dict, Truths, Loss_info)


def metric(model,Dataset,Loss_info, **kwargs):

    ''' For now i am not sure what to do'''
    Loss_info = merge_parameters(Loss_info, kwargs)
    keys       = Loss_info.get('keys', ['Chi_0','Rp','T0'])
    Debug_Mode = Loss_info.get('Debug_Mode', False)
    
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
    

    def __init__(self, in_main_channels =(2,), pixel_embedding_size = 32, latent_space_size = 32,N_dense_nodes = 64,Train_Type = 'Both',**kwargs):
        super(Model_Autoencoder_TimeFit, self).__init__()
        
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

        self.Geometry_OutWeights = torch.tensor([1.0,1.0,1.0])
            
    def forward(self,Graph,Aux):
        if self.Debug_Mode and self.training:
            raise NotImplementedError('Debug Mode is not implemented')
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
            Pix_Data[i,1:N_pix_in_event[i]+1,:] = torch.stack([event['chi_is'], event['time']], dim=-1).to(device)
            Pix_Data[i,0                    ,:] = torch.tensor([event['station_chii'], event['station_time']], device=device) 

        pix_embedding = self.pix_process(Pix_Data) # (B, N_pix, pixel_embedding_size)
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
            
            max_chis = torch.tensor([event['chi_is'].max() if event['chi_is'].numel() > 0 else 0 for event in Graph], device=device).unsqueeze(1) # (B, 1)
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






class LatentPixelBlock(nn.Module):
    def __init__(self, pixel_dim, latent_dim, hidden_dim, n_heads):
        super().__init__()

        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads

        # projections
        self.to_q = nn.Linear(latent_dim, hidden_dim)
        self.to_k = nn.Linear(pixel_dim , hidden_dim)
        self.to_v = nn.Linear(pixel_dim , hidden_dim)

        # output projection after attention pooling
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # latent update
        self.update = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # gating (stability)
        self.gate = nn.Linear(latent_dim + hidden_dim, latent_dim)

        # norms
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_pixel  = nn.LayerNorm(pixel_dim)

    def forward(self, X, mask, Z):
        """
        X   : (B, P, F)
        mask: (B, P)  (1 = valid, 0 = padding)
        Z   : (B, D)
        """
        
        B, P, _ = X.shape

        # ---- normalize ----   # not sure about this, maybe it will help with stability, maybe it will ruin things
        X = self.norm_pixel(X)
        Z = self.norm_latent(Z)

        # ---- projections ----
        Q = self.to_q(Z)                      # (B,    Hdim)
        K = self.to_k(X)                      # (B, P, Hdim)
        V = self.to_v(X)                      # (B, P, Hdim)

        # ---- reshape to heads ----
        Q = Q.view(B   , self.n_heads, self.head_dim)         # (B, H, d)
        K = K.view(B, P, self.n_heads, self.head_dim)         # (B, P, H, d)
        V = V.view(B, P, self.n_heads, self.head_dim)

        K = K.permute(0, 2, 1, 3)   # (B, H, P, d)
        V = V.permute(0, 2, 1, 3)   # (B, H, P, d)

        # ---- masked attention pooling ----
        # (B, H, P)
        attn_logits = (Q.unsqueeze(2) * K).sum(-1) / torch.sqrt(torch.tensor(self.head_dim))
        
        mask = mask.unsqueeze(1)  # (B, 1, P)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn_logits, dim=-1)

        pooled = (attn.unsqueeze(-1) * V).sum(dim=2)   # (B, H, d)

        pooled = pooled.reshape(B, -1)                 # (B, Hdim)
        pooled = self.to_out(pooled)

        # ---- latent update ----
        combined = torch.cat([Z, pooled], dim=-1)  # Pix Data + Latent Space Proj
        dz = self.update(combined)
        gate = torch.sigmoid(self.gate(combined))   # update control, i might be interested in how much this is magnitudinally, to decide when to stop iteration.  gate and dz by themselves are not useful, need to have combined to make decisions on iteration exit.
        Z_change = gate * dz
        Z_new = Z + Z_change

        # return attn, pooled, Z_new
        return {'Z_new': Z_new, 'attn': attn, 'pooled': pooled,'Z_change': Z_change}



class Model_Autoencoder_TimeFit_Recursive(nn.Module):
    Name = 'Model_Autoencoder_TimeFit_Recursive'
    Description = '''Model will try to figure out a state representation of the shower time profile
    Reconstruction of geometry is done from this state
    Training to be done in two steps
    1. train the autoencoder to reconstruct the time profile
        -  to do this, model will observe the data and try to produce 10 new values from the profile
    2. train the decoder that will use the latent state to reconstruct the geometry
        
    '''
    

    def __init__(self, in_main_channels =(3,), pixel_embedding_size = 32, latent_space_size = 32,N_dense_nodes = 64,N_heads=4,Train_Type = 'Both',Exit_Early = False,**kwargs):
        super(Model_Autoencoder_TimeFit_Recursive, self).__init__()
        
        self.kwargs = kwargs

        self.Pix_Features = in_main_channels[0] + 1 -1 # +1 Time Diffs -1 Charge # TODO: this will be wrong, need to check if i am using charge or not, if i am using extra trace features
        self.Train_Type   = Train_Type
        
        # assert Train_Type in ['Profile','Geometry','Both'], 'Train_Type should be one of Profile, Geometry, Both'
        assert Train_Type == 'Both', 'This model is tuned to be trained on Geometry and Profile at the same time'
        
        self.Debug_Mode = kwargs.get('Debug_Mode', False)
        
        self.max_latent_iterations     = kwargs.get('max_latent_iterations'    , 10  )
        self.latent_space_change_scale = kwargs.get('latent_space_change_scale', 0.01)
        self.allow_early_exit = Exit_Early
        print(f'    Model_Init - setting allow_early_exit to {self.allow_early_exit} and max_latent_iterations to {self.max_latent_iterations}')
        self.iteration_weights = torch.linspace(0.2,1.0, steps=self.max_latent_iterations) if Exit_Early else torch.ones(self.max_latent_iterations)
        self.iteration_weights = self.iteration_weights / self.iteration_weights.sum() # Normalize the weights, sum to 1

        ###################
        self.initial_latent_space = nn.Parameter(torch.zeros(latent_space_size)) # (latent_space_size,)
        self.update_latent_space = LatentPixelBlock(pixel_dim=self.Pix_Features, latent_dim=latent_space_size, hidden_dim=pixel_embedding_size, n_heads=N_heads)

        self.produce_time_profile = nn.Sequential(
            nn.Linear(latent_space_size + 1, N_dense_nodes), # +1 for the u coordinate
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 1) # predict t and q
        )

        self.produce_geometry = nn.Sequential(
            nn.Linear(latent_space_size, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, N_dense_nodes),
            nn.LeakyReLU(),
            nn.Linear(N_dense_nodes, 3) # predict Chi_0* and log_Rp and T0
        )
        

        self.Geometry_OutWeights = kwargs.get('OutWeights', torch.tensor([1.0,1.0,1.0]))
        print(f'    Model_Init - Geometry_OutWeights set to {self.Geometry_OutWeights}')







    def forward(self,Graph,Aux):
        if self.Debug_Mode and self.training:
            raise NotImplementedError('Debug Mode is not implemented')
            return self.debug_forward(Graph,Aux)
        else:
            return self.clean_forward(Graph,Aux)
        
    def clean_forward(self,Graph,Aux):

        device = self.initial_latent_space.device
        
        # ---- Unpack the graph data into a pixel space representation ----
        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        Time_norm_by_event = torch.tensor([event['norm_min_time'] for event in Graph], device=device).unsqueeze(1) # (B, 1)

        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        
        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features-1], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:2] = torch.stack([event['chi_is'], event['time']], dim=-1).to(device)
            Pix_Data[i,0                    ,:2] = torch.tensor([event['station_chii'], event['station_time']], device=device) 

        
        # ---- Initial conditions and preprocessing for the recursive latent space update ----
        max_chis = torch.tensor([event['chi_is'].max() if event['chi_is'].numel() > 0 else 0 for event in Graph], device=device).unsqueeze(1) # (B, 1)


        pix_diffs = torch.zeros([N_Events, N_pix_in_event.max()+1, 1],device=device)
        pix_space    = torch.cat([Pix_Data, pix_diffs], dim=-1) # (B, N_pix, Pix_Features + 1)
        
        latent_space = self.initial_latent_space.unsqueeze(0).expand(N_Events, -1) # (B, latent_space_size)

        input_chi_is = Pix_Data[:,:,0].unsqueeze(-1) # (B, N_pix, 1)
        input_times  = Pix_Data[:,:,1].unsqueeze(-1) # (B, N_pix, 1)
        
        N_iter = 0
        Acc_Loss = 0
        Acc_Loss_track = []
        # ---- Recursive Latent Space Update Loop ----
        while N_iter < self.max_latent_iterations:

            # ----  Update Latent Space ----
            update_out = self.update_latent_space(pix_space, Mask, latent_space)
            
            latent_space  = update_out['Z_new']
            latent_change = update_out['Z_change']


            # ---- Make a reconstruction on initial pixel space ----
            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, input_chi_is.shape[1], -1)

            time_profile_input = torch.cat([latent_space_expanded, input_chi_is], dim=-1) # (B, 10, latent_space_size + 1)
            B,S,D = time_profile_input.shape
            
            Reco_Time = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 1)
            Reco_Time = Reco_Time.reshape(B,S,1)
            
            pix_diffs = (Reco_Time - input_times).detach()
            pix_space = pix_space.clone() # (B, N_pix, Pix_Features + 1)
            pix_space[:,:,-1] = pix_diffs.squeeze(-1)
            

            # ---- Calculate the reconstruction loss for this iteration, make sure to use mask ----

            Acc_Loss += self.iteration_weights[N_iter] * F.mse_loss(Reco_Time[Mask], input_times[Mask])
            Acc_Loss_track.append(F.mse_loss(Reco_Time, input_times).item())

            # ---- Early Exit Check ----
            N_iter += 1
            if self.allow_early_exit and (N_iter > 1) and (latent_change.norm() < latent_space.norm()*self.latent_space_change_scale): # There is a thought to use dz and gate to make decisions on iteration exit. 
                break

        
        # ---- Predict the Time Profile and Geometry from the final latent space ----
        
        test_chis = torch.linspace(0, 1, steps=10, device=device).unsqueeze(0).repeat(N_Events, 1) # (B, 10) # between 0 and 1
        test_chis = (test_chis * max_chis).unsqueeze(-1) # (B, 10) # between 0 and chi_0

        latent_space_expanded = latent_space.unsqueeze(1).expand(-1, test_chis.shape[1], -1)

        time_profile_input = torch.cat([latent_space_expanded, test_chis], dim=-1) # (B, 10, latent_space_size + 1)
        B,S,D = time_profile_input.shape
        
        time_profile = self.produce_time_profile(time_profile_input.reshape(B*S,D)) # (B* 10, 2)
        time_profile = time_profile.reshape(B,S,1)

        geometry = self.produce_geometry(latent_space)*self.Geometry_OutWeights.to(device) # (B, 3)

        # ---- Hard Coded Geometry Unnormalisation ----
        # geometry[:,0]  = torch.arccos(geometry[:,0]) # Chi_0 is between 0 and pi
        # geometry[:,1]  = geometry[:,1] * 5e4 # scale roughly 0-1 
        # geometry[:,2]  = geometry[:,2] * 1e5 # scale roughly -2.5-1 0.5ish 
        
        intermideate_chi_0 = torch.tanh(geometry[:,0]) 
        
        geom_out = torch.stack([
            intermideate_chi_0,
            geometry[:,1],
            geometry[:,2]
        ], dim=-1)

        return {
            'time_profile': time_profile, # (B, 10, 1)
            'test_chi_is' : test_chis   , # (B, 10, 1)
            'geometry'    : geom_out    , # (B, 3)
            'Acc_Loss'    : Acc_Loss    , # (scalar) - this is the loss of the time profile reconstruction at each iteration, can be used to analyze convergence and maybe make decisions on iteration exit in future versions
            'Time_Norm'   : Time_norm_by_event.squeeze(-1),

            # ---- Unused for training, might be useful in analysis ----
            'N_iter'      : N_iter      , # (scalar)
            'latent_space': latent_space,  # (B, latent_space_size) - this is the final latent space representation after the recursive updates, can be used for analysis and maybe for other purposes in future versions
            'Acc_Loss_track': Acc_Loss_track # (list of scalars) - this tracks the reconstruction loss at each iteration, can be useful for analyzing convergence behavior
        }
    


class Multi_headedAttention(nn.Module): # Same as LatentPixelBlock but without the detailed output, might reduce this further later # TODO
    def __init__(self, pixel_dim, latent_dim, hidden_dim, n_heads):
        super().__init__()

        assert hidden_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads

        # projections
        self.to_q = nn.Linear(latent_dim, hidden_dim)
        self.to_k = nn.Linear(pixel_dim , hidden_dim)
        self.to_v = nn.Linear(pixel_dim , hidden_dim)

        # output projection after attention pooling
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

        # latent update
        self.update = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # gating (stability)
        self.gate = nn.Linear(latent_dim + hidden_dim, latent_dim)

        # norms
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_pixel  = nn.LayerNorm(pixel_dim)

    def forward(self, X, mask, Z):
        """
        X   : (B, P, F)
        mask: (B, P)  (1 = valid, 0 = padding)
        Z   : (B, D)
        """
        
        B, P, _ = X.shape

        # ---- normalize ----   # not sure about this, maybe it will help with stability, maybe it will ruin things
        X = self.norm_pixel(X)
        Z = self.norm_latent(Z)

        # ---- projections ----
        Q = self.to_q(Z)                      # (B,    Hdim)
        K = self.to_k(X)                      # (B, P, Hdim)
        V = self.to_v(X)                      # (B, P, Hdim)

        # ---- reshape to heads ----
        Q = Q.view(B   , self.n_heads, self.head_dim)         # (B, H, d)
        K = K.view(B, P, self.n_heads, self.head_dim)         # (B, P, H, d)
        V = V.view(B, P, self.n_heads, self.head_dim)

        K = K.permute(0, 2, 1, 3)   # (B, H, P, d)
        V = V.permute(0, 2, 1, 3)   # (B, H, P, d)

        # ---- masked attention pooling ----
        # (B, H, P)
        attn_logits = (Q.unsqueeze(2) * K).sum(-1) / torch.sqrt(torch.tensor(self.head_dim))
        
        mask = mask.unsqueeze(1)  # (B, 1, P)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn_logits, dim=-1)

        pooled = (attn.unsqueeze(-1) * V).sum(dim=2)   # (B, H, d)

        pooled = pooled.reshape(B, -1)                 # (B, Hdim)
        pooled = self.to_out(pooled)

        # ---- latent update ----
        combined = torch.cat([Z, pooled], dim=-1)  # Pix Data + Latent Space Proj
        dz = self.update(combined)
        gate = torch.sigmoid(self.gate(combined))   # update control, i might be interested in how much this is magnitudinally, to decide when to stop iteration.  gate and dz by themselves are not useful, need to have combined to make decisions on iteration exit.
        Z_change = gate * dz
        
        return Z_change



class Model_Latent_Iterator_only_geometry(nn.Module):
    Name = 'Model_Latent_Iterator'
    Description = '''
    Model that updates iteratively updates pixel space and latent embedding (residual style)
    No early Exit
    Only geometry preds on the first attempt, might add the time profile preds in the future.
    No Early exit is supposed
    '''
    ArgumentsList = [
        'in_main_channels', 'pixel_embedding_size', 'latent_space_size', 'N_dense_nodes', 'N_heads', 'Train_Type',
        'Debug_Mode', 'max_latent_iterations', 'Early_Exit',
        'init_pix_embedding_activation'  , 'init_pix_embedding_dropout'   ,
        'pix_space_updater_activation'   ,'pix_space_updater_dropout'     ,
        'latent_space_updater_activation', 'latent_space_updater_dropout' ,
        'geometry_predictor_activation'  , 'geometry_predictor_dropout'   ,
        'OutWeights'
    ]
    def __init__(self,
                 in_main_channels =(3,), 
                 pixel_embedding_size = 32, 
                 latent_space_size = 32,
                 N_dense_nodes = 64,
                 N_heads = 4,
                 Train_Type = 'Geometry',
                 **kwargs):
        super(Model_Latent_Iterator_only_geometry, self).__init__()

        # ---- Parse Arguments ---- # 
        self.Pix_Features = in_main_channels[0] - 1 # Not doing charge for now
        self.Train_Type   = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)
        self.max_latent_iterations = kwargs.get('max_latent_iterations', 5)
        self.Early_Exit = kwargs.get('Early_Exit', False)

        # ---- Make architecture based assertions ---- #
        assert self.Train_Type == 'Geometry', 'This model is trained only on Geometry'
        assert self.Early_Exit == False     , 'Early Exit is assumed False for this model'
        


        # ---- Define the architecture ---- # 

        # Initial Pixel Embedding
        # (B,N_pix, Pix_features) ->  (B,N_pix, pixel_embedding_size)
        init_pix_embedding_activation = kwargs.get('init_pix_embedding_activation', nn.ReLU)
        self.init_pix_embedding = nn.Sequential(
                                                nn.Linear(self.Pix_Features, pixel_embedding_size),
                                                init_pix_embedding_activation(),
                                                nn.Dropout(kwargs.get('init_pix_embedding_dropout', 0.2)),
                                                nn.Linear(pixel_embedding_size, pixel_embedding_size),
        )   
        


        # Initial Latent Space
        self.initial_latent_space = nn.Parameter(torch.randn(latent_space_size)) # Using gaussian, will train to something else i guess

        # Layer Norms here
        self.pix_embed_norm    = nn.LayerNorm(pixel_embedding_size)
        self.latent_space_norm = nn.LayerNorm(latent_space_size)


        # pix_space_updater
        # (B, N_pix , pixel_embedding_size + latent_space_size) -> (B, N_pix, pixel_embedding_size)
        pix_space_updater_activation = kwargs.get('pix_space_updater_activation', nn.ReLU)
        self.pix_space_updater = nn.Sequential(
                                               nn.Linear(pixel_embedding_size + latent_space_size, pixel_embedding_size),
                                               pix_space_updater_activation(),
                                               nn.Dropout(kwargs.get('pix_space_updater_dropout', 0.2)),
                                               nn.Linear(pixel_embedding_size, pixel_embedding_size)
        )


        # Headed Attention block
        # (B, N_pix, pixel_embedding_size) -> (B, latent_space_size)
        # Have mask of valid pixels to 
        self.MHA_latent_update = Multi_headedAttention(pixel_dim=pixel_embedding_size, latent_dim=latent_space_size, hidden_dim=pixel_embedding_size, n_heads=N_heads)


        # latent_space_updater
        # (B, latent_space_size) -> (B, latent_space_size)
        latent_space_activation = kwargs.get('latent_space_updater_activation', nn.ReLU)
        self.latent_space_updater = nn.Sequential(
                                                nn.Linear(latent_space_size, 2*N_dense_nodes),
                                                latent_space_activation(),
                                                nn.Dropout(kwargs.get('latent_space_updater_dropout', 0.2)),
                                                nn.Linear(2*N_dense_nodes, latent_space_size)
        )

        # Geometry Predictor
        # (B, latent_space_size) -> (B, N_out)
        geometry_activation = kwargs.get('geometry_predictor_activation', nn.ReLU)
        PredStyle = kwargs.get('PredStyle','SimpleGeometry')
        if PredStyle in ['SimpleGeometry','Unnormed_Chi_0']:
            N_out = 3
            self.OutWeights = kwargs.get('OutWeights', torch.tensor([1.0,1.0,1.0]))
        
        if PredStyle == 'Double_Chi_0':
            N_out = 4
            self.OutWeights = kwargs.get('OutWeights', torch.tensor([1.0,1.0,1.0,0.0]))
            assert self.OutWeights.shape[0] == 4, 'OutWeights should have 4 elements for Double_Chi_0 PredStyle'
        
        self.produce_geometry = nn.Sequential(
                                              nn.Linear(latent_space_size, N_dense_nodes),
                                              geometry_activation(),
                                              nn.Dropout(kwargs.get('geometry_predictor_dropout', 0.2)),
                                              nn.Linear(N_dense_nodes, N_out) # predict Chi_0* and log_Rp and T0
        )

        
        


    def forward(self,Graph,Aux):
        device = self.initial_latent_space.device
        
        # ---- Unpack the graph data into a pixel space representation ----
        N_Events = len(Graph)
        N_pix_in_event = torch.tensor(list(map(len,map(lambda x: x['chi_is'],Graph))),device=device).int()
        Time_norm_by_event = torch.tensor([event['norm_min_time'] for event in Graph], device=device).unsqueeze(1) # (B, 1)

        Mask = torch.arange(N_pix_in_event.max()+1,device=device).unsqueeze(0) < (N_pix_in_event+1).unsqueeze(1) # (B, N_pix)

        
        Pix_Data = torch.zeros([N_Events, N_pix_in_event.max()+1, self.Pix_Features], device=device) # +1 for station data
        for i,event in enumerate(Graph):
            Pix_Data[i,1:N_pix_in_event[i]+1,:2] = torch.stack([event['chi_is'], event['time']], dim=-1).to(device)
            Pix_Data[i,0                    ,:2] = torch.tensor([event['station_chii'], event['station_time']], device=device) 


        # ---- initial embeddings ---- #
        latent_space = self.initial_latent_space.unsqueeze(0).expand(N_Events, -1) # (B, latent_space_size)
        
        pix_embedding = self.init_pix_embedding(Pix_Data) # (B, N_pix, pixel_embedding_size)

        # ---- Recursive Latent Space Update Loop ----
        for iteration in range(self.max_latent_iterations):

            # Update Pixel Space with current Latent Space
            latent_space_expanded = latent_space.unsqueeze(1).expand(-1, pix_embedding.shape[1], -1)
            pix_update_input = torch.cat([pix_embedding, latent_space_expanded], dim=-1)

            pix_embedding_update = self.pix_space_updater(pix_update_input)
            pix_embedding = pix_embedding + pix_embedding_update


            # Update Latent Space with current Pixel Space
            latent_space_update = self.MHA_latent_update(pix_embedding, Mask, latent_space)
            latent_space = latent_space + latent_space_update

            # Further update the latent space with a feedforward network # Going to skip this bit for now dont think i need to do this # TODO
            latent_space_update = self.latent_space_updater(latent_space)
            latent_space = latent_space + latent_space_update

        # ---- Predict Geometry from the final latent space ----
        geometry = self.produce_geometry(latent_space)*self.OutWeights.to(device)

        return {
            'geometry'    : geometry    , # (B, N_out)
            'Time_Norm'   : Time_norm_by_event.squeeze(-1),
        }




class Loss_class():
    def __init__(self, Loss_info, **kwargs):
        self.Loss_info = merge_parameters(Loss_info, kwargs)
        # Training Info
        self.Train_Type = self.Loss_info.get('Train_Type', 'Geometry')
        self.OutWeights = self.Loss_info.get('OutWeights', torch.tensor([1.0,1.0,1.0]))
        self.Debug_Mode = self.Loss_info.get('Debug_Mode', False)

        assert self.Train_Type == 'Geometry', 'This loss class is only implemented for Geometry training'
        if self.Debug_Mode: print(f'    Loss Class initialized with Train Type: {self.Train_Type} and Debug Mode: {self.Debug_Mode} which is NOT IMPLEMENTED')

        # Loss Info
        self.keys         = Loss_info.get('keys'         , ['Chi_0','Rp','T0'])
        self.ReturnTensor = Loss_info.get('ReturnTensor' , True  ) 




    def __call__(self, Pred, Truth, Extra_Loss_Info, **kwargs):
        
        Geometry_Pred = Pred['geometry']
        Time_Norm     = Pred['Time_Norm']
        device = Geometry_Pred.device

        
        if Extra_Loss_Info.get('PredStyle','SimpleGeometry') == 'Double_Chi_0':
            losses = {}
            losses['T0'] = F.mse_loss(Geometry_Pred[:,3]+ Time_Norm *1e9/1e7, Truth[:,2].to(device))*self.OutWeights[3] # Pixel Time Shift in units of ns
            losses['Rp'] = F.mse_loss(Geometry_Pred[:,2], Truth[:,1].to(device))*self.OutWeights[2]

            Truth_Chi_0_cos = Truth[:,0].to(device)
            Truth_Chi_0_sin = torch.sqrt(1 - Truth_Chi_0_cos**2 + 1e-8) # Add small epsilon for numerical stability
            losses['Chi_0'] = F.mse_loss(Geometry_Pred[:,0], Truth_Chi_0_cos)*self.OutWeights[0] + F.mse_loss(Geometry_Pred[:,1], Truth_Chi_0_sin)*self.OutWeights[1]
        if Extra_Loss_Info.get('PredStyle','SimpleGeometry') == 'Unnormed_Chi_0':
            losses = {}
            losses['Chi_0'] = F.mse_loss(Geometry_Pred[:,0], torch.acos(Truth[:,0].to(device)))     *self.OutWeights[0]
            losses['Rp'] = F.mse_loss(Geometry_Pred[:,1], Truth[:,1].to(device))                    *self.OutWeights[1]
            losses['T0'] = F.mse_loss(Geometry_Pred[:,2]+ Time_Norm *1e9/1e7, Truth[:,2].to(device))*self.OutWeights[2]

        else:
            losses = {}
            for i,key in enumerate(self.keys):
                if key == 'T0':
                    losses[key] = F.mse_loss(Geometry_Pred[:,i]+ Time_Norm *1e9/1e7, Truth[:,i].to(device))*self.OutWeights[i] # Pixel Time Shift in units of ns 
                if key == 'Chi_0':
                    losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i].to(device))*self.OutWeights[i]
                else:
                    losses[key] = F.mse_loss(Geometry_Pred[:,i], Truth[:,i].to(device))*self.OutWeights[i]
            



        losses['Total'] = sum(losses[key] for key in self.keys)
        if not self.ReturnTensor: losses = {key:loss.item() for key,loss in losses.items()}
        
        return losses
    

class Validate_class():
    '''
    Class Variant of the validate call
    Takes model, Dataset, Loss Function, device, keys
    Dataset is defined as ProcessingDatasetContainer in the Dataset2.py
    keys are to be used in the loss function
    BatchSize to change in case it doesnt fit into memory
    Returns the average loss
    '''
    def __init__(self, Loss_Info, **kwargs):
        self.Loss_Info = merge_parameters(Loss_Info, kwargs)
        self.Train_Type = self.Loss_Info.get('Train_Type', 'Geometry')
        # No Need to do assertions, assuming they will be tripped by loss long before validation is called


    def __call__(self, model, Dataset, Loss, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)

        if Debug_Mode: print('################ VALIDATION ################')

        # make sure the Dataset State is Val
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Dataset.BatchSize = len(Dataset) // 256 
        # Dataset.BatchSize = 1 if Debug_Mode else BatchSize  # CARE THIS THING AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        
        model.eval()

        

        Geometry_preds = []
        Truths         = []
        Time_Norms     = []
        
        with torch.no_grad():
            for i,(_, BatchMains, BatchAux, BatchTruth, _)  in enumerate(Dataset):
                Model_out = model(BatchMains,BatchAux)
                
                Geometry_preds.append(Model_out['geometry'].to('cpu'))
                
                Time_Norms.append(Model_out.get('Time_Norm', torch.zeros(BatchTruth.shape[0])).to('cpu'))
                Truths.append(BatchTruth.to('cpu'))

                if Debug_Mode: break

            
            
            
            Geometry_preds = torch.cat(Geometry_preds,dim=0)
            Time_Norms = torch.cat(Time_Norms,dim=0)
            Truths     = torch.cat(Truths    ,dim=0)
            
        # Return Batch Size to old value
        Dataset.BatchSize = TrainingBatchSize
        
        Loss_inp_dict = {
            'geometry'    : Geometry_preds,
            'Time_Norm'  : Time_Norms
        }
        return Loss(Loss_inp_dict, Truths, Loss_info)


class Metric_class():
    def __init__(self, Loss_Info, **kwargs):
        self.Loss_Info = merge_parameters(Loss_Info, kwargs)
        self.Train_Type = self.Loss_Info.get('Train_Type', 'Geometry')
        # No Need to do assertions, assuming they will be tripped by loss long before validation is called


    def __call__(self, model, Dataset, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)


        if Debug_Mode: print('################ METRICS ################')

        # ---- Setup ----
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Dataset.BatchSize = len(Dataset) //256

        model.eval()
        
        Geometry_preds = []
        Truths         = []
        Time_Norms     = []
        # ---- Calculate
        
        with torch.no_grad():
            for i,(_, BatchMains, BatchAux, BatchTruth, _)  in enumerate(Dataset):
                Model_out = model(BatchMains,BatchAux)
                
                Geometry_preds.append(Model_out['geometry'].to('cpu'))
                Time_Norms.append(Model_out.get('Time_Norm', torch.zeros(BatchTruth.shape[0])).to('cpu'))
                Truths.append(BatchTruth.to('cpu'))

                if Debug_Mode: break
            
            Geometry_preds = torch.cat(Geometry_preds,dim=0)
            Time_Norms = torch.cat(Time_Norms,dim=0)
            Truths     = torch.cat(Truths    ,dim=0)


        # ---- Metrics Calculation ----
        # Unnormalise Data
        if Extra_Loss_Info.get('PredStyle','SimpleGeometry') in ['Default','SimpleGeometry']:
            Geometry_preds[:,0] = torch.clip(Geometry_preds[:,0], -0.999, 0.999) # Clip to avoid numerical issues with arccos
        if Extra_Loss_Info.get('PredStyle','SimpleGeometry') == 'Unnormed_Chi_0':
            Geometry_preds[:,0] = torch.acos(Geometry_preds[:,0])
        if Extra_Loss_Info.get('PredStyle','SimpleGeometry') == 'Double_Chi_0':
            Geometry_preds_Chi_0 = torch.cos(torch.atan2(Geometry_preds[:,0], Geometry_preds[:,1]))
            Geometry_preds = torch.stack([Geometry_preds_Chi_0, Geometry_preds[:,2], Geometry_preds[:,3]], dim=-1)


        Geometry_preds = Dataset.Unnormalise_Truth(Geometry_preds)
        Truths         = Dataset.Unnormalise_Truth(Truths)

        Geometry_preds[:,2] = Geometry_preds[:,2] + Time_Norms *1e9/1e7


        Metric_Style = Loss_info.get('Metric_Style', 'Default')
        if Metric_Style in ['Default','Percentrile68']:
            Metrics = {}
            for i,key in enumerate(['Chi_0','Rp','T0']):
                Metrics[key] = torch.quantile(torch.abs(Geometry_preds[:,i] - Truths[:,i]), 0.68).item() 
                
        Dataset.BatchSize = TrainingBatchSize
        return Metrics
