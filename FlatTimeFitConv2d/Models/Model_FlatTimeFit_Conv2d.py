# Importing the libraries
from pdb import main
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


class Conv_Skip_Block_2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=nn.ReLU,
        dropout=0.2,
        kernel_size=3,
        padding=1,
    ):
        super(Conv_Skip_Block_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

        # Project residual if channel dims differ
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += residual
        return self.activation(out)


def main_from_graph(batch, H=75, W=1000):
    """
    batch: list of dicts with keys 'indices' (N,2) and 'values' (N,)
    returns: (B, 1, H, W) float tensor
    """
    B = len(batch)
    out = torch.zeros(B, 1, H, W)
    for i, sample in enumerate(batch):
        idx = sample['indices']    # (N, 2) — row/col indices
        val = sample['values']     # (N,)
        out[i, 0, idx[:, 0], idx[:, 1]] = torch.as_tensor(val, dtype=torch.float32)
    return out

class Model_FlatTimeFit_Conv2d(nn.Module):
    Name = 'Model_FlatTimeFit_Conv2d'
    Description = '''
    Flat Time Fit Tensor input
    Conv2d style architecture
    '''
    ArgumentsList = [
        'OutWeights',
        'kernel_size', 'N_kernels', 'stride', 'padding',
        'pool_kernel_size', 'pool_stride',
        'Train_Type', 'Debug_Mode', 'head_dropout'

    ]

    def __init__(self,
                 in_main_channels=(1,),
                 N_kernels=16,
                 kernel_size=(5,19),
                 stride=(1,1),
                 padding=(2,9),
                 pool_kernel_size=(2, 4),
                 pool_stride=(2, 4),
                 Train_Type='Geometry',
                 OutWeights = torch.tensor([1.0,0.0,0.0]),
                 **kwargs):

        super(Model_FlatTimeFit_Conv2d, self).__init__()

        # ---- Parse Arguments ---- #
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)

        # ---- Make architecture based assertions ---- #
        assert self.Train_Type == 'Geometry', \
            'This model is trained only on Geometry'

        # Input shape: (B, 1, 75, 1000)
        in_ch = in_main_channels[0]  # 1
        assert in_ch == 1, 'Input should have 1 channel for Flat Time Fit Tensor (after merging chi and time)'

        # ---- Stage 1: 1 → N_kernels ---- #
        # After pool: (B, N_kernels, 37, 250)
        self.stage1 = nn.Sequential(
            Conv_Skip_Block_2d(in_ch,       N_kernels,     kernel_size=kernel_size, padding=padding),
            Conv_Skip_Block_2d(N_kernels,   N_kernels,     kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )

        # ---- Stage 2: N_kernels → 2*N_kernels ---- #
        # After pool: (B, 2*N_kernels, 18, 62)
        self.stage2 = nn.Sequential(
            Conv_Skip_Block_2d(N_kernels,   N_kernels * 2, kernel_size=kernel_size, padding=padding),
            Conv_Skip_Block_2d(N_kernels*2, N_kernels * 2, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )

        # ---- Stage 3: 2*N_kernels → 4*N_kernels ---- #
        # After pool: (B, 4*N_kernels, ~9, ~31) — collapsed by AdaptivePool
        self.stage3 = nn.Sequential(
            Conv_Skip_Block_2d(N_kernels*2, N_kernels * 4, kernel_size=kernel_size, padding=padding),
            Conv_Skip_Block_2d(N_kernels*4, N_kernels * 4, kernel_size=kernel_size, padding=padding),
            nn.AdaptiveAvgPool2d((1, 1)),   # (B, 4*N_kernels, 1, 1)
        )

        # ---- Classification Head ---- #
        feat_dim = N_kernels * 4   # 64 when N_kernels=16
        head_hidden = feat_dim * 2 # 128

        self.head = nn.Sequential(
            nn.Flatten(),                             # (B, feat_dim)
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(kwargs.get('head_dropout', 0.3)),
            nn.Linear(head_hidden, 3),                # (B, 3)
        )

        self.OutWeights = OutWeights

        if self.Debug_Mode:
            print(f'[{self.Name}] feat_dim={feat_dim}, head_hidden={head_hidden}')

    def forward(self, graph, aux):
        x = main_from_graph(graph)
        device = next(self.parameters()).device

        x = x.to(device)

        # x: (B, 1, 75, 1000)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        x = x*self.OutWeights.to(device)
        
        out = {
            'geometry': x, # (B, 3)
            'Time_Norm': torch.tensor([event['norm_min_time'] for event in graph], device=x.device) # (B,)
        }
        return out


# ---- Quick shape check ---- #
if __name__ == '__main__':
    model = Model_FlatTimeFit_Conv2d(Debug_Mode=True)
    dummy = torch.zeros(4, 1, 75, 1000)
    print(f'Input shape: {dummy.shape}')  # torch.Size([4, 1, 75, 1000])
    out = model(dummy)
    print(f'Output shape: {out.shape}')  # torch.Size([4, 3])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params:,}')
