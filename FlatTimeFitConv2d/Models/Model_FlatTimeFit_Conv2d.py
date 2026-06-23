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

        self.Done_Operational_PrintOut = False


    def __call__(self, Pred, Truth, Extra_Loss_Info, **kwargs):
        
        Geometry_Pred = Pred['geometry']
        Time_Norm     = Pred['Time_Norm']
        device = Geometry_Pred.device

        PredStyle = Extra_Loss_Info.get('PredStyle','SimpleGeometry')

        if not self.Done_Operational_PrintOut:
            print()
            print(f'    Loss Class called with PredStyle: {PredStyle}')
            print(f'    Out Weights: {self.OutWeights}')
            print()

            self.Done_Operational_PrintOut = True
        
        if PredStyle == 'Double_Chi_0':
            losses = {}
            losses['T0'] = F.mse_loss(Geometry_Pred[:,3]+ Time_Norm *1e9/1e7, Truth[:,2].to(device))*self.OutWeights[3] # Pixel Time Shift in units of ns
            losses['Rp'] = F.mse_loss(Geometry_Pred[:,2], Truth[:,1].to(device))*self.OutWeights[2]

            Truth_Chi_0_cos = Truth[:,0].to(device)
            Truth_Chi_0_sin = torch.sqrt(1 - Truth_Chi_0_cos**2 + 1e-8) # Add small epsilon for numerical stability
            losses['Chi_0'] = F.mse_loss(Geometry_Pred[:,0], Truth_Chi_0_cos)*self.OutWeights[0] + F.mse_loss(Geometry_Pred[:,1], Truth_Chi_0_sin)*self.OutWeights[1]
        
        elif PredStyle == 'Unnormed_Chi_0':
            losses = {}
            losses['Chi_0'] = F.mse_loss(Geometry_Pred[:,0], torch.acos(Truth[:,0].to(device)))        *self.OutWeights[0]
            losses['Rp']    = F.mse_loss(Geometry_Pred[:,1], Truth[:,1].to(device))                    *self.OutWeights[1]
            losses['T0']    = F.mse_loss(Geometry_Pred[:,2]+ Time_Norm *1e9/1e7, Truth[:,2].to(device))*self.OutWeights[2]
        
        elif PredStyle == 'TanOfChi_0':
            losses = {}
            losses['Chi_0'] = F.mse_loss(Geometry_Pred[:,0], torch.tan(torch.acos(Truth[:,0].to(device))/2.0))*self.OutWeights[0]
            losses['Rp']    = F.mse_loss(Geometry_Pred[:,1], Truth[:,1].to(device))                    *self.OutWeights[1]
            losses['T0']    = F.mse_loss(Geometry_Pred[:,2]+ Time_Norm *1e9/1e7, Truth[:,2].to(device))*self.OutWeights[2]
        
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
        self.Hit_Batch_Size_Limit = 1e99

    def __call__(self, model, Dataset, Loss, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)

        if Debug_Mode: print('################ VALIDATION ################')

        # make sure the Dataset State is Val
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Dataset.BatchSize = min(len(Dataset) // 256 ,self.Hit_Batch_Size_Limit)
        # Dataset.BatchSize = 1 if Debug_Mode else BatchSize  # CARE THIS THING AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        
        Success = False
        while not Success:
            try :
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

                    
                    
                    
            except Exception as e:
                if 'CUDA out of memory' in str(e):
                    Dataset.BatchSize = Dataset.BatchSize // 2
                    self.Hit_Batch_Size_Limit = Dataset.BatchSize
                    print(f'Hit CUDA Memory Limit, reducing batch size to {Dataset.BatchSize}')
                else:
                    raise e
            finally:
                Success = True

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
        self.Hit_Batch_Size_Limit = 1e99
        # No Need to do assertions, assuming they will be tripped by loss long before validation is called
        self.Done_Operational_PrintOut = False

    def __call__(self, model, Dataset, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)

        if Debug_Mode: print('################ METRICS ################')

        # ---- Setup ----
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Success = False
        while not Success:
            try:
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
            except Exception as e:
                if 'CUDA out of memory' in str(e):
                    Dataset.BatchSize = Dataset.BatchSize // 2
                    self.Hit_Batch_Size_Limit = Dataset.BatchSize
                    print(f'Hit CUDA Memory Limit, reducing batch size to {Dataset.BatchSize}')
                else:
                    raise e
            finally:
                Success = True

            Geometry_preds = torch.cat(Geometry_preds,dim=0)
            Time_Norms = torch.cat(Time_Norms,dim=0)
            Truths     = torch.cat(Truths    ,dim=0)

        PredStyle = Extra_Loss_Info.get('PredStyle','SimpleGeometry')
        if not self.Done_Operational_PrintOut:
            print(f'    Metric Class called with PredStyle: {PredStyle}')
            self.Done_Operational_PrintOut = True
        
        # ---- Metrics Calculation ----
        # Unnormalise Data
        if PredStyle in ['Default','SimpleGeometry']:
            Geometry_preds[:,0] = torch.clip(Geometry_preds[:,0], -0.999, 0.999) # Clip to avoid numerical issues with arccos
        
        elif PredStyle == 'Unnormed_Chi_0':
            Geometry_preds[:,0] = torch.acos(Geometry_preds[:,0])
        
        elif PredStyle == 'Double_Chi_0':
            Geometry_preds_Chi_0 = torch.cos(torch.atan2(Geometry_preds[:,1], Geometry_preds[:,0]))
            Geometry_preds = torch.stack([Geometry_preds_Chi_0, Geometry_preds[:,2], Geometry_preds[:,3]], dim=-1)
        
        elif PredStyle == 'TanOfChi_0':
            Geometry_preds[:,0] = torch.clip(torch.atan(Geometry_preds[:,0])*2, -0.999, 0.999) # Clip to avoid numerical issues with arccos
            Geometry_preds[:,0] = torch.cos(Geometry_preds[:,0])

        else: 
            raise ValueError(f'Unknown PredStyle: {PredStyle}')




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
                 Main_Folding = (1,1),
                 **kwargs):

        super(Model_FlatTimeFit_Conv2d, self).__init__()

        # ---- Parse Arguments ---- #
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)
        self.Main_Folding = Main_Folding

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
        # Need to sum x over the dimensions according to the folding. 
        # For example, if Main_Folding = (2,2), we need to sum every 2 neighbouring pixels in both dimensions
        # We can pad the input to make it divisible by the folding dimensions, and then reshape and sum
        # Sum over the folding dimensions
        if self.Main_Folding != (1,1):
            fold_h, fold_w = self.Main_Folding
            B, C, H, W = x.shape
            pad_h = (fold_h - H % fold_h) % fold_h
            pad_w = (fold_w - W % fold_w) % fold_w
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
            x = x.view(B, C, H_padded // fold_h, fold_h, W_padded // fold_w, fold_w)
            x = x.sum(dim=[3,5])

        


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




class Model_FlatTimeFit_Conv2d_Simple(nn.Module):
    Name = 'Model_FlatTimeFit_Conv2d_Simple'
    Description = '''
    Flat Time Fit Tensor input
    Conv2d style architecture
    '''
    ArgumentsList = [
        'OutWeights',
        'kernel_size', 'N_kernels', 'stride', 'padding',
        'pool_kernel_size', 'pool_stride',
        'Train_Type', 'Debug_Mode', 'head_dropout'
        'PredStyle',

    ]

    def __init__(self,
                 in_main_channels=(1,),
                 N_dense_nodes = 128,
                 N_kernels=16,
                 kernel_size=(5,19),
                 stride=(1,1),
                 padding=(2,9),
                 pool_kernel_size=(1, 1), # 1,1 for no pooling
                 pool_stride=(1,1),
                 Train_Type='Geometry',
                 OutWeights = torch.tensor([1.0,1.0,0.0]),
                 Main_Folding = (1,1),
                 PredStyle = 'SimpleGeometry',
                 **kwargs):

        super(Model_FlatTimeFit_Conv2d_Simple, self).__init__()

        # ---- Parse Arguments ---- #
        self.Train_Type = Train_Type
        self.Debug_Mode = kwargs.get('Debug_Mode', False)
        self.Main_Folding = Main_Folding

        # ---- Make architecture based assertions ---- #
        assert self.Train_Type == 'Geometry', \
            'This model is trained only on Geometry'

        # Input shape: (B, 1, 75, 1000)
        in_ch = in_main_channels[0]  # 1
        assert in_ch == 1, 'Input should have 1 channel for Flat Time Fit Tensor (after merging chi and time)'

        # Conv Layers
        self.Conv_1 = nn.Conv2d(in_ch, N_kernels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.Conv_2 = nn.Conv2d(N_kernels, N_kernels*2, kernel_size=kernel_size, padding=padding, stride=stride)
        self.Conv_3 = nn.Conv2d(N_kernels*2, N_kernels*4, kernel_size=kernel_size, padding=padding, stride=stride)

        self.pool_1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.Conv_4 = nn.Conv2d(N_kernels*4, N_kernels*4, kernel_size=kernel_size, padding=padding, stride=stride)
        self.Conv_5 = nn.Conv2d(N_kernels*4, N_kernels*4, kernel_size=kernel_size, padding=padding, stride=stride)
        self.Conv_6 = nn.Conv2d(N_kernels*4, N_kernels*4, kernel_size=kernel_size, padding=padding, stride=stride)

        self.pool_2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Fully Connected Layers
        # Figure out the shape of the layers via just pushing a dummy tensor through the conv layers, this is easier than doing the math and less error prone
        x = torch.zeros(1, in_ch, 75, 1000)
        if self.Main_Folding != (1,1):
            fold_h, fold_w = self.Main_Folding
            B, C, H, W = x.shape
            pad_h = (fold_h - H % fold_h) % fold_h
            pad_w = (fold_w - W % fold_w) % fold_w
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
            x = x.view(B, C, H_padded // fold_h, fold_h, W_padded // fold_w, fold_w)
            x = x.sum(dim=[3,5])
        print(f'Model Init - After Main Folding Model Expects an input of shape: {x.shape}') # This is the input shape for the conv layers
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)
        x = self.pool_1(x)
        x = self.Conv_4(x)
        x = self.Conv_5(x)
        x = self.Conv_6(x)
        x = self.pool_2(x)
        print(f'Model init - Shape after conv layers: {x.shape}') # This is the input shape for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten

        feat_dim = x.shape[1]
        print(f'Model init - Feature dimension after conv layers: {feat_dim}') # This is the input dimension for the fully connected layers


        N_out = len(OutWeights)
        self.FC_1 = nn.Linear(feat_dim, N_dense_nodes)
        self.FC_2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.FC_3 = nn.Linear(N_dense_nodes, N_out)

        self.Conv_Activation = nn.LeakyReLU()
        self.FC_Activation   = nn.LeakyReLU()

        self.OutWeights = OutWeights

    def forward(self, graph, aux):
        x = main_from_graph(graph)
        device = next(self.parameters()).device

        x = x.to(device)

        # Need to sum x over the dimensions according to the folding. 
        # For example, if Main_Folding = (2,2), we need to sum every 2 neighbouring pixels in both dimensions
        # We can pad the input to make it divisible by the folding dimensions, and then reshape and sum
        # Sum over the folding dimensions

        if self.Main_Folding != (1,1):
            fold_h, fold_w = self.Main_Folding
            B, C, H, W = x.shape
            pad_h = (fold_h - H % fold_h) % fold_h
            pad_w = (fold_w - W % fold_w) % fold_w
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
            x = x.view(B, C, H_padded // fold_h, fold_h, W_padded // fold_w, fold_w)
            x = x.sum(dim=[3,5])

        
        # x: (B, 1, 75, 1000)
        x = self.Conv_Activation(self.Conv_1(x))
        x = self.Conv_Activation(self.Conv_2(x))
        x = self.Conv_Activation(self.Conv_3(x))
        
        x = self.pool_1(x)
        
        x = self.Conv_Activation(self.Conv_4(x))
        x = self.Conv_Activation(self.Conv_5(x))
        x = self.Conv_Activation(self.Conv_6(x))
        
        x = self.pool_2(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.FC_Activation(self.FC_1(x))
        x = self.FC_Activation(self.FC_2(x))
        x = self.FC_3(x)
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
