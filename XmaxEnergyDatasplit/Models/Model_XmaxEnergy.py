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

def merge_parameters(PriorityParameters, SecondaryParameters):
    return {**SecondaryParameters, **PriorityParameters} # This overwrites the Secondary if there is same Priority key

# Define the Loss Function

class LossBuffer:
    def __init__(self, maxlen=100):
        self.buffer = torch.zeros(maxlen)
        self.maxlen = maxlen
        self.ptr    = 0
        self.count  = 0

    def push(self, loss_val):
        self.buffer[self.ptr] = loss_val
        self.ptr              = (self.ptr + 1) % self.maxlen
        self.count            = min(self.count + 1, self.maxlen)

    def push_batch(self, vals):
        vals = vals.detach().reshape(-1)
        n = vals.numel()
        if n >= self.maxlen: raise ValueError( f"Number of values is too large for the buffer with n={n} and buffer size = {self.maxlen}") 
        
        end = self.ptr + n

        if end <= self.maxlen:
            self.buffer[self.ptr:end] = vals
        else:
            first = self.maxlen - self.ptr
            self.buffer[self.ptr:] = vals[:first]
            self.buffer[:end - self.maxlen] = vals[first:]
        self.ptr   = end % self.maxlen
        self.count = min(self.count + n, self.maxlen)

    def stats(self):
        valid = self.buffer[:self.count]
        return valid.mean(), valid.std()

    def percentile(self, q):
        valid = self.buffer[:self.count]
        return torch.quantile(valid, q / 100.0)

    def is_ready(self, min_samples=320):
        return self.count >= min_samples
    

class Loss_class():

    def __init__(self, Loss_info, **kwargs):
        self.Loss_info = merge_parameters(Loss_info, kwargs)
        # Training Info
        self.OutWeights = self.Loss_info.get('OutWeights', torch.tensor([1.0,1.0]))
        self.Debug_Mode = self.Loss_info.get('Debug_Mode', False)

        # Loss Info
        self.keys         = Loss_info.get('keys'         , ['Xmax','LogE'])
        self.ReturnTensor = Loss_info.get('ReturnTensor' , True  )

        self.Loss_Buffers = {key: LossBuffer(maxlen=3200) for key, out_weight in zip(self.keys, self.OutWeights) if out_weight > 0} # 3200 is default 100 batches
        print(f'Initialized Loss Buffers for keys: {[key for key, out_weight in zip(self.keys, self.OutWeights) if out_weight > 0]}')

        self.Use_Buffer_Masking = self.Loss_info.get('Use_Buffer_Masking',False)
        
        self.Huber_Deltas = self.Loss_info.get('Huber_Deltas',[1,0.25]) # Somewhat based on distributions ive alredy seen
        
        self.Gate_Lambda_Percentile = self.Loss_info.get('Gate_Lambda_Percentile', 60)

    def __call__(self, Pred, Truth, Extra_Loss_Info, **kwargs):
        if isinstance(Pred, dict) and 'Rejection' in Pred:
            return self._call_gated(Pred, Truth, Extra_Loss_Info, **kwargs)

        if isinstance(Pred, dict):
            Pred = Pred['Pred']
        is_validation = kwargs.get('validation_call',False)

        if self.Use_Buffer_Masking:
            Truth = Truth.to(Pred.device)
            losses = {}
            for i, key in enumerate(self.keys):
                if self.OutWeights[i] == 0:
                    losses[key] = torch.tensor(0.0, device=Pred.device)
                else:
                    i_buffer = self.Loss_Buffers[key]
                    i_Loss = F.mse_loss(Pred[:,i], Truth[:,i], reduction='none')
                    if i_buffer.is_ready():
                        mean, std = i_buffer.stats()
                        upper_bound = mean + 3*std
                        i_Loss_mask = (i_Loss < upper_bound)
                        if i_Loss_mask.any():
                            i_Loss = i_Loss[i_Loss_mask]
                    i_Loss = i_Loss.mean()
                    if not is_validation : i_buffer.push(i_Loss)

                    losses[key] = i_Loss
            losses['Total'] = sum(losses[key] for key in self.keys)
            if not self.ReturnTensor: losses = {key:loss.item() for key,loss in losses.items()}
            return losses

        else:
            Truth = Truth.to(Pred.device)
            losses = {}
            for i, key in enumerate(self.keys):
                if self.OutWeights[i] == 0:
                    losses[key] = torch.tensor(0.0, device=Pred.device)
                else:
                    losses[key] = F.huber_loss(Pred[:,i], Truth[:,i], delta=self.Huber_Deltas[i])
            losses['Total'] = sum(losses[key] for key in self.keys)
            if not self.ReturnTensor: losses = {key:loss.item() for key,loss in losses.items()}
            return losses

    def _call_gated(self, Pred, Truth, Extra_Loss_Info, **kwargs):
        Extra_Loss_Info = merge_parameters(Extra_Loss_Info, kwargs)
        Output = Pred['Pred']
        Gate   = Pred['Rejection']
        Truth  = Truth.to(Output.device)
        losses = {}
        is_validation = Extra_Loss_Info.get('validation_call',False)
        use_adaptable_Lambda  = Extra_Loss_Info.get('Use_Adaptable_Lambda', False)
                
        for i, key in enumerate(self.keys):
            if self.OutWeights[i] == 0:
                losses[key] = torch.tensor(0.0, device=Output.device)
                continue

            i_Loss = F.huber_loss(Output[:,i], Truth[:,i], delta=self.Huber_Deltas[i], reduction='none')
            i_gate = Gate[:,i]
            
            # Check if gates are all at one or zero
            if torch.all(i_gate< 0.01):
                
                # print(f"Warning: All gates for key {key} are {'open' if torch.all(i_gate> 0.99) else 'closed'}, skipping gating for this key. batch = {Extra_Loss_Info.get('Batch','Unknown')}")
                print(f"Warning: All gates for key {key} are closed, skipping gating for this key. batch = {Extra_Loss_Info.get('Batch','Unknown')}")


            i_buffer = self.Loss_Buffers[key]
            if i_buffer.is_ready():
                Lambda = i_buffer.percentile(self.Gate_Lambda_Percentile)
            else:
                Lambda = i_Loss.detach().mean()

            # Flat Lambda
            Max_Lambda = 0.04
            Lambda = min(Lambda, Max_Lambda) if use_adaptable_Lambda else Max_Lambda
            
            if not is_validation: i_buffer.push_batch(i_Loss)
            if Extra_Loss_Info.get('Epoch',0) > 1:
                losses[key] = (i_gate * i_Loss + (1 - i_gate) * Lambda).mean()
            else:
                losses[key] = i_Loss.mean() # Warmup period of one epoch

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

    def __call__(self, model, Dataset, Loss, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)

        if Debug_Mode: print('################ VALIDATION ################')

        # make sure the Dataset State is Val
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Dataset.BatchSize = len(Dataset) // 256

        model.eval()

        Preds  = []
        Truths = []
        Gates  = []
        gated  = None

        with torch.no_grad():
            for i, (_, BatchMains, BatchAux, BatchTruth, _) in enumerate(Dataset):
                model_out = model(BatchMains, BatchAux)

                if gated is None:
                    if isinstance(model_out, dict):
                        if 'Rejection' in model_out:
                            gated = True
                        else:
                            raise NotImplementedError(f"Found Model Out Dict, without Rejection, not implemented")
                    else:
                        gated = False

                if not gated:
                    Preds .append(model_out.to('cpu'))
                else:
                    Preds .append(model_out['Pred']     .to('cpu'))
                    Gates .append(model_out['Rejection'].to('cpu'))

                Truths.append(BatchTruth.to('cpu'))

            Preds  = torch.cat(Preds ,dim=0)
            Truths = torch.cat(Truths,dim=0)
            if gated:
                Gates = torch.cat(Gates,dim=0)

        # Return Batch Size to old value
        Dataset.BatchSize = TrainingBatchSize

        Loss_in = Preds if not gated else {'Pred': Preds, 'Rejection': Gates}

        return Loss(Loss_in, Truths, Loss_info,validation_call = True)


def metric(model,Dataset,device,keys=['Xmax','LogE'],BatchSize = 256):
    '''    Takes model, Dataset, Loss Function, device, keys
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


class Metric_class:
    def __init__(self, Loss_Info, **kwargs):
        self.Loss_Info = merge_parameters(Loss_Info, kwargs)
        self.Gate_Threshold = self.Loss_Info.get('Gate_Threshold', 0.5)

    def __call__(self, model, Dataset, Extra_Loss_Info, **kwargs):
        Loss_info = merge_parameters(Extra_Loss_Info, kwargs)
        Debug_Mode = Loss_info.get('Debug_Mode', False)
        Gate_Threshold = Loss_info.get('Gate_Threshold', self.Gate_Threshold)

        keys = Loss_info.get('keys', ['Xmax','LogE'])

        if Debug_Mode: print('################ METRICS ################')

        # ---- Setup ---- #
        Dataset.State = 'Val'
        TrainingBatchSize = Dataset.BatchSize
        Dataset.BatchSize = len(Dataset) // 256

        model.eval()

        Preds  = []
        Truths = []
        Gates  = []
        gated  = None

        # ---- Calculate ---- #
        with torch.no_grad():
            for _, BatchMains, BatchAux, BatchTruth, _ in Dataset:
                model_out = model(BatchMains, BatchAux)

                if gated is None:
                    if isinstance(model_out, dict):
                        if 'Rejection' in model_out:
                            gated = True
                        else:
                            raise NotImplementedError(f"Found Model Out Dict, without Rejection, not implemented")
                    else:
                        gated = False

                if not gated:
                    Preds.append(model_out.to('cpu'))
                else:
                    Preds.append(model_out['Pred'].to('cpu'))
                    Gates.append(model_out['Rejection'].to('cpu'))

                Truths.append(BatchTruth.to('cpu'))

        Preds  = torch.cat(Preds, dim=0).cpu()
        Truths = torch.cat(Truths, dim=0).cpu()
        if gated:
            Gates = torch.cat(Gates, dim=0).cpu()

        Preds  = Dataset.Unnormalise_Truth(Preds)
        Truths = Dataset.Unnormalise_Truth(Truths)

        metrics = {}
        metric_units = []
        Units = Dataset.Truth_Units
        for i, key in enumerate(keys):
            if gated:
                mask = Gates[:, i] >= Gate_Threshold
                if mask.sum() == 0:
                    metrics[key] = torch.tensor(float('nan'))
                    continue
                pred_i, truth_i = Preds[mask, i], Truths[mask, i]
            else:
                pred_i, truth_i = Preds[:, i], Truths[:, i]

            if Units[i] == 'rad':
                AngDiv = torch.atan2(torch.sin(pred_i - truth_i), torch.cos(pred_i - truth_i))
                metrics[key] = torch.quantile(torch.abs(AngDiv), 0.68)
                metric_units.append('rad')
            elif Units[i] == 'deg':
                AngDiv = torch.atan2(
                    torch.sin(torch.deg2rad(pred_i - truth_i)),
                    torch.cos(torch.deg2rad(pred_i - truth_i))
                )
                metrics[key] = torch.quantile(torch.abs(AngDiv), 0.68) * 180 / torch.pi
                metric_units.append('deg')
            else:
                metrics[key] = torch.quantile(torch.abs(pred_i - truth_i), 0.68)
                metric_units.append(Units[i])
            
            if gated:
                metrics[f'{key}_GatedFraction'] = 1.0 - mask.float().mean()
                metric_units.append('')


        # Return Batch Size to old value
        Dataset.BatchSize = TrainingBatchSize
        metrics['Units'] = metric_units
        return metrics


####################################################################################################################

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


class Model_XmaxEnergy_Conv3d(nn.Module):
    Name = 'Model_XmaxEnergy_Conv3d'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_Conv3d, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout3d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)


        self.conv0 = nn.Conv3d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,1,0) , stride = 1) # Out=> (N, N_kernels, 40, 20, 20)
        self.Conv1 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv2 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv3 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)

        self.Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = kwargs.get('OutWeights', torch.tensor([1.0,1.0]))

        # Best State Dict
        self.best_state_dict = None

    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        
        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100 ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it.

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        # Main = Main.unfold(1,10,10)
        # Main = Main.sum(-1)
        # Main = Main[:,:80,:,:].unsqueeze(1).to(device)
       
       # Dont need to rebin this, because our dimensions are already small
       # Just take the first 40 bins
        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1

        # Process the Data
        Main = self.Conv_Activation(self.conv0(Main))
        # Main = self.Conv_Dropout(Main)
        Main = self.Conv1(Main)
        Main = self.Conv2(Main)
        Main = self.Conv3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return Output

    def save_best_state_dict(self):
        del self.best_state_dict
        self.best_state_dict = self.state_dict()


class Model_XmaxEnergy_Conv3d_withRejection(nn.Module):
    Name = 'Model_XmaxEnergy_Conv3d_withRejection'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution

    Rejection is a classified output to decide wether model rejects calculation
    '''

    def __init__(self,in_main_channels = (1,), N_kernels = 32, N_dense_nodes = 128, **kwargs):
        # only one Main is expected
        assert len(in_main_channels) == 1, 'Only one Main Channel is expected'
        in_main_channels = in_main_channels[0]
        self.kwargs = kwargs
        dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.2
        
        super(Model_XmaxEnergy_Conv3d_withRejection, self).__init__()

        # Activation Function
        self.Conv_Activation  = nn.LeakyReLU()
        self.Dense_Activation = nn.ReLU()
        self.Angle_Activation = nn.Tanh()
        self.Conv_Dropout     = nn.Dropout3d(dropout)
        self.Dense_Dropout    = nn.Dropout(dropout)
        self.Rejection_Activation = nn.Sigmoid()


        self.conv0 = nn.Conv3d(in_channels=in_main_channels, out_channels=N_kernels, kernel_size=3, padding = (1,1,0) , stride = 1) # Out=> (N, N_kernels, 40, 20, 20)
        self.Conv1 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv2 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)
        self.Conv3 = Conv_Skip_Block_3d(in_channels=N_kernels, N_kernels=N_kernels, activation_function=self.Conv_Activation, kernel_size=(3,3,3), padding=(1,1,1), dropout=dropout)

        self.Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Dense3 = nn.Linear(N_dense_nodes, N_dense_nodes)

        self.Xmax1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Xmax2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Xmax3 = nn.Linear(N_dense_nodes//2,1)

        self.Energy1 = nn.Linear(N_dense_nodes,N_dense_nodes)
        self.Energy2 = nn.Linear(N_dense_nodes,N_dense_nodes//2)
        self.Energy3 = nn.Linear(N_dense_nodes//2,1)


        self.OutWeights = kwargs.get('OutWeights', torch.tensor([1.0,1.0]))

        # Rejection Module
        self.Rejection_Dense1 = nn.Linear(N_kernels*40*20*20, N_dense_nodes)
        self.Rejection_Dense2 = nn.Linear(N_dense_nodes, N_dense_nodes)
        self.Rejection_Dense3 = nn.Linear(N_dense_nodes, 2)


        # Best State Dict
        self.best_state_dict = None

    def forward(self,Graph,Aux=None):
        
        # Unpack the Graph Datata to Main
        device = self.Dense1.weight.device
        NEvents = len(Graph)
        
        TraceMain = torch.zeros(NEvents,40   ,20,22)
        StartMain = torch.zeros(NEvents,1    ,20,22)
        Main      = torch.zeros(NEvents,2100 ,20,22) 
        # Have to allocate this massive tenosr to avoid memory issues
        # Maybe there is a better way to do this, but for now i cannot think of it.

        N_pixels_in_event = torch.tensor(list(map(len,map(lambda x : x[0], Graph)))).int()
        EventIndices      = torch.repeat_interleave(torch.arange(NEvents),N_pixels_in_event).int()

        Traces = torch.cat(list(map(lambda x : x[0], Graph)))
        Xs     = torch.cat(list(map(lambda x : x[1], Graph)))
        Ys     = torch.cat(list(map(lambda x : x[2], Graph)))
        Pstart = torch.cat(list(map(lambda x : x[3], Graph)))

        TraceMain[EventIndices,:,Xs,Ys] = Traces
        StartMain[EventIndices,0,Xs,Ys] = Pstart

        indices = (torch.arange(40).reshape(1,-1,1,1)+StartMain).long()
        Main.scatter_(1,indices,TraceMain)
        
        # Rebin Main
        # Main = Main.unfold(1,10,10)
        # Main = Main.sum(-1)
        # Main = Main[:,:80,:,:].unsqueeze(1).to(device)
       
       # Dont need to rebin this, because our dimensions are already small
       # Just take the first 40 bins
        Main = Main[:,:40,:,:].unsqueeze(1).to(device)
        Main[torch.isnan(Main)] = -1

        # Process the Data
        Main = self.Conv_Activation(self.conv0(Main))
        # Main = self.Conv_Dropout(Main)
        Main = self.Conv1(Main)
        Main = self.Conv2(Main)
        Main = self.Conv3(Main)

        # Flatten the output
        Main = Main.view(Main.shape[0],-1)
        # Calculate Rejection before Main is Mutated
        Rejection = self.Dense_Activation(self.Rejection_Dense1(Main))
        Rejection = self.Dense_Activation(self.Rejection_Dense2(Rejection))
        Rejection = self.Rejection_Activation(self.Rejection_Dense3(Rejection))

        # Dense and Output Layers
        Main = self.Dense_Activation(self.Dense1(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense2(Main))
        Main = self.Dense_Dropout(Main)
        Main = self.Dense_Activation(self.Dense3(Main))

        Xmax   = self.Dense_Activation(self.Xmax1(Main))
        Xmax   = self.Dense_Activation(self.Xmax2(Xmax))
        Xmax   = self.Xmax3(Xmax)

        Energy = self.Dense_Activation(self.Energy1(Main))
        Energy = self.Dense_Activation(self.Energy2(Energy))
        Energy = self.Energy3(Energy)

        Output = torch.cat([Xmax,Energy],dim=1)* self.OutWeights.to(device)
        return {'Pred':Output,'Rejection':Rejection}

    def save_best_state_dict(self):
        del self.best_state_dict
        self.best_state_dict = self.state_dict()

class Model_XmaxEnergy_Conv3d_fromRejection(Model_XmaxEnergy_Conv3d_withRejection):
    Name = 'Model_XmaxEnergy_Conv3d_fromRejection'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution

    Rejection is a classified output to decide wether model rejects calculation

    This model takes the Rejection filtered dataset for training
    This model still has Rejection - see if it will prioritise outliers

    '''


class Model_XmaxEnergy_Conv3d_withRejection_ForSpoofedDataset(Model_XmaxEnergy_Conv3d_withRejection):
    Name = 'Model_XmaxEnergy_Conv3d_withRejection_ForSpoofedDataset'
    Description = '''
    Convolutional Neural Network for SDP Reconstruction
    Uses standard Conv3d Layers
    Reconstruction is done for one telescope
    No pooling is done, because it ruins the resolution

    Rejection is a classified output to decide wether model rejects calculation

    This model is specifically designed for the Energy Spoofed Dataset, where the Energy range is stretched
    '''


####################################################################################################################

