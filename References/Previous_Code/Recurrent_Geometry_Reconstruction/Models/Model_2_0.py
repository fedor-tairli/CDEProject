# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
# from   torch_geometric.nn import GCNConv, TAGConv,GATConv
# from   torch_geometric.nn.pool import global_mean_pool, global_max_pool
# from   torch_geometric.utils import add_self_loops
import numpy as np
import os

import sys
import paths
Paths = paths.load_ProjectPaths(paths.get_caller_dir())
# add code path into the path
sys.path.append(Paths.code_path)

from TrainingModule import IterateInBatches




# Define the model

class Model_2_0_Donor(nn.Module):
    # Info
    Name = 'Model_2_0_Donor'
    Description = '''
    Part of the Model_2_X series
    The Donor Model

    Predict Without Aux Data
    Transplant the LSTM Weights into the Recipient Model
    '''

    def __init__(self,in_channels = 2,in_AuxData = 3,n_dense_nodes = 64,n_LSTM_nodes = 32,LSTM_layers = 3,dropout_prob=0,Dtype = torch.float32):
        LSTM_dropout_prob = 0.5*dropout_prob
        super(Model_2_0_Donor, self).__init__()
        self.LSTM1 = nn.LSTM(in_channels, n_LSTM_nodes, LSTM_layers, batch_first=True,bidirectional=True,dropout=LSTM_dropout_prob,dtype=Dtype)
        self.LSTM2 = nn.LSTM(n_LSTM_nodes*2, n_LSTM_nodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        # Dense Layers
        self.Chi0Dense1 = nn.Linear(n_LSTM_nodes,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(n_dense_nodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(n_LSTM_nodes,n_dense_nodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(n_dense_nodes,1,dtype=Dtype)

        # Activation and dropout
        self.DenseActivation  = nn.LeakyReLU()
        self.dropout          = nn.Dropout(dropout_prob)

    def forward(self, Traces, AuxData):
        # LSTM 
        out, _ = self.LSTM1(Traces)
        out, _ = self.LSTM2(out)

        # Take the last output
        out = out[:,-1,:]
        
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)

class Model_2_0_Recipient(nn.Module):
    # Info
    Name = 'Model_2_0_Recipient'
    Description = '''
    Part of the Model_2_X series
    The Recipient Model

    Predict Without Aux Data
    Transplant the LSTM Weights from the Donor Model
    '''

    def __init__(self,in_channels = 2,in_AuxData = 3,n_dense_nodes = 128,n_LSTM_nodes = 32,LSTM_layers = 3,dropout_prob=0,Dtype = torch.float32):
        LSTM_dropout_prob = 0.5*dropout_prob
        super(Model_2_0_Recipient, self).__init__()
        self.LSTM1 = nn.LSTM(in_channels, n_LSTM_nodes, LSTM_layers, batch_first=True,bidirectional=True,dropout=LSTM_dropout_prob,dtype=Dtype)
        self.LSTM2 = nn.LSTM(n_LSTM_nodes*2, n_LSTM_nodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        # Aux Analysis
        self.AuxDense1 = nn.Linear(in_AuxData+n_LSTM_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense3 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)

        # Dense Layers
        self.Chi0Dense1 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(n_dense_nodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(n_dense_nodes,1,dtype=Dtype)

        # Activation
        self.DenseActivation  = nn.LeakyReLU()
        self.dropout          = nn.Dropout(dropout_prob)

    def forward(self, Traces, AuxData):
        # LSTM 
        out, _ = self.LSTM1(Traces)
        out, _ = self.LSTM2(out)

        # Take the last output
        out = out[:,-1,:]
        out = torch.cat((out,AuxData),dim=1)

        # Aux Analysis
        out = self.DenseActivation(self.AuxDense1(out))
        out = self.DenseActivation(self.AuxDense2(out))
        out = self.DenseActivation(self.AuxDense3(out))
        out = self.dropout(out)
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)

    def TransplantWeights(self,DonorModel:Model_2_0_Donor):
        '''
        Transplant the weights from the Donor Model
        '''
        # LSTM only needs being transplanted here:
        self.LSTM1.load_state_dict(DonorModel.LSTM1.state_dict())
        self.LSTM2.load_state_dict(DonorModel.LSTM2.state_dict())




class Model_2_1_Donor(nn.Module):
    # Info
    Name = 'Model_2_1_Donor'
    Description = '''
    Part of the Model_2_X series
    The Donor Model

    Predict Without Aux Data
    Transplant the LSTM Weights into the Recipient Model
    This model uses the hidden states to pass information also
    '''

    def __init__(self,in_channels = 2,in_AuxData = 3,n_dense_nodes = 64,n_LSTM_nodes = 16,LSTM_layers = 3,dropout_prob=0,Dtype = torch.float32):
        LSTM_dropout_prob = 0.5*dropout_prob        
        super(Model_2_1_Donor, self).__init__()
        self.LSTM1 = nn.LSTM(in_channels, n_LSTM_nodes, LSTM_layers, batch_first=True,bidirectional=True,dropout=LSTM_dropout_prob,dtype=Dtype)
        self.LSTM2 = nn.LSTM(n_LSTM_nodes*2, n_LSTM_nodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        self.LSTMDEnse1 = nn.Linear(n_LSTM_nodes*3  ,n_dense_nodes//2,dtype=Dtype)
        self.LSTMDEnse2 = nn.Linear(n_dense_nodes//2,n_dense_nodes//2,dtype=Dtype)
        self.LSTMDEnse3 = nn.Linear(n_dense_nodes//2,n_dense_nodes//2,dtype=Dtype)

        # Dense Layers
        self.Chi0Dense1 = nn.Linear(n_dense_nodes//2,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(n_dense_nodes   ,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(n_dense_nodes   ,1          ,dtype=Dtype)

        self.RpDense1 = nn.Linear(n_dense_nodes//2,n_dense_nodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(n_dense_nodes   ,n_dense_nodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(n_dense_nodes   ,1          ,dtype=Dtype)

        # Activation and Dropout
        self.DenseActivation  = nn.LeakyReLU()
        self.Dropout          = nn.Dropout(dropout_prob)

    def forward(self, Traces, AuxData):
        # LSTM 
        out, _ = self.LSTM1(Traces)
        out, Hidden = self.LSTM2(out)
        # Take the last output
        out = out[:,-1,:]
        # Concatenate the Hidden States
        out = torch.cat((out,Hidden[0].squeeze(),Hidden[1].squeeze()),dim=1)
        out = self.DenseActivation(self.LSTMDEnse1(out))
        out = self.DenseActivation(self.LSTMDEnse2(out))
        out = self.DenseActivation(self.LSTMDEnse3(out))

        out = self.Dropout(out)

        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)

class Model_2_1_Recipient(nn.Module):
    # Info
    Name = 'Model_2_1_Recipient'
    Description = '''
    Part of the Model_2_X series
    The Recipient Model

    Predict Without Aux Data
    Transplant the LSTM Weights from the Donor Model
    This model uses the hidden states to pass information also
    '''

    def __init__(self,in_channels = 2,in_AuxData = 3,n_dense_nodes = 64,n_LSTM_nodes = 16,LSTM_layers = 3,dropout_prob=0,Dtype = torch.float32):
        LSTM_dropout_prob = 0.5*dropout_prob
        super(Model_2_1_Recipient, self).__init__()
        self.LSTM1 = nn.LSTM(in_channels, n_LSTM_nodes, LSTM_layers, batch_first=True,bidirectional=True,dropout=LSTM_dropout_prob,dtype=Dtype)
        self.LSTM2 = nn.LSTM(n_LSTM_nodes*2, n_LSTM_nodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        self.LSTMDEnse1 = nn.Linear(n_LSTM_nodes*3  ,n_dense_nodes//2,dtype=Dtype)
        self.LSTMDEnse2 = nn.Linear(n_dense_nodes//2,n_dense_nodes//2,dtype=Dtype)
        self.LSTMDEnse3 = nn.Linear(n_dense_nodes//2,n_dense_nodes//2,dtype=Dtype)

        # Aux Analysis
        self.AuxDense1 = nn.Linear(in_AuxData+n_dense_nodes//2,n_dense_nodes,dtype=Dtype)
        self.AuxDense2 = nn.Linear(n_dense_nodes              ,n_dense_nodes,dtype=Dtype)
        self.AuxDense3 = nn.Linear(n_dense_nodes              ,n_dense_nodes,dtype=Dtype)

        # Dense Layers
        self.Chi0Dense1 = nn.Linear(n_dense_nodes            ,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(n_dense_nodes            ,n_dense_nodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(n_dense_nodes            ,1          ,dtype=Dtype)

        self.RpDense1 = nn.Linear(n_dense_nodes              ,n_dense_nodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(n_dense_nodes              ,n_dense_nodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(n_dense_nodes              ,1          ,dtype=Dtype)

        # Activation and dropout
        self.DenseActivation  = nn.LeakyReLU()
        self.Dropout          = nn.Dropout(dropout_prob)

        # Final Activations
        self.Chi0Activation   = nn.Tanh()
        # No Activation for Rp
        

    def forward(self, Traces, AuxData):
        # LSTM 
        out, _ = self.LSTM1(Traces)
        out, Hidden = self.LSTM2(out)
        # Take the last output
        out = out[:,-1,:]
        # Concatenate the Hidden States
        out = torch.cat((out,Hidden[0].squeeze(),Hidden[1].squeeze()),dim=1)
        out = self.DenseActivation(self.LSTMDEnse1(out))
        out = self.DenseActivation(self.LSTMDEnse2(out))
        out = self.DenseActivation(self.LSTMDEnse3(out))
        out = self.Dropout(out)
        # Aux Analysis
        out = torch.cat((out,AuxData),dim=1)
        out = self.DenseActivation(self.AuxDense1(out))
        out = self.DenseActivation(self.AuxDense2(out))
        out = self.DenseActivation(self.AuxDense3(out))
        out = self.Dropout(out)
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Activation (self.Chi0Dense3(Chi0))

        Rp   = self.DenseActivation(self.RpDense1(out))
        Rp   = self.DenseActivation(self.RpDense2(Rp ))
        Rp   =                      self.RpDense3(Rp )
        
        return torch.cat((Chi0,Rp),dim=1)

    def TransplantWeights(self,DonorModel:Model_2_0_Donor):
        '''
        Transplant the weights from the Donor Model
        '''
        # LSTM only needs being transplanted here:
        self.LSTM1.load_state_dict(DonorModel.LSTM1.state_dict())
        self.LSTM2.load_state_dict(DonorModel.LSTM2.state_dict())



# Define the Loss Function
    
def LossMSE(Pred,Truth):

    '''
    Takes Truth,Pred in form -> [CosPhi, SinPhi, CosTheta, SinTheta]
    Calculates MSE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    # Chi0
    Chi0Truth = Truth[0]
    Chi0Pred = Pred[0]
    Chi0Loss = F.mse_loss(Chi0Pred,Chi0Truth)

    # Rp
    RpTruth = Truth[1]
    RpPred = Pred[1]
    RpLoss = F.mse_loss(RpPred,RpTruth)

    
    # Sum up
    Total_Loss = Chi0Loss + RpLoss
    return Total_Loss,Chi0Loss,RpLoss

def LossMAE(Pred,Truth):

    '''
    Takes Truth,Pred in form -> [CosPhi, SinPhi, CosTheta, SinTheta]
    Calculates MAE Loss, outputs Total Loss, Phi Loss, Theta Loss
    '''
    assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
    # Chi0
    Chi0Truth = Truth[0]
    Chi0Pred = Pred[0]
    Chi0Loss = F.l1_loss(Chi0Pred,Chi0Truth)

    # Rp
    RpTruth = Truth[1]
    RpPred = Pred[1]
    RpLoss = F.l1_loss(RpPred,RpTruth)

    
    # Sum up
    Total_Loss = Chi0Loss + RpLoss
    return Total_Loss,Chi0Loss,RpLoss

def LossHuber(Pred,Truth):
    
        '''
        Takes Truth,Pred in form -> [CosPhi, SinPhi, CosTheta, SinTheta]
        Calculates Huber Loss, outputs Total Loss, Phi Loss, Theta Loss
        '''
        assert Pred.shape == Truth.shape, f'Pred Shape: {Pred.shape}, Truth Shape: {Truth.shape} not equal'
        # Chi0
        Chi0Truth = Truth[0]
        Chi0Pred = Pred[0]
        Chi0Loss = F.smooth_l1_loss(Chi0Pred,Chi0Truth)
    
        # Rp
        RpTruth = Truth[1]
        RpPred = Pred[1]
        RpLoss = F.smooth_l1_loss(RpPred,RpTruth)
    
        
        # Sum up
        Total_Loss = Chi0Loss + RpLoss
        return Total_Loss,Chi0Loss,RpLoss
# Set Default Loss in case I forget this bit in Training
Loss = LossMSE 

# Validation metrics fpr printout
def Percentile68(Truths,Predictions):
    return torch.quantile(torch.abs(Truths-Predictions),0.68)

# Define the Validation Function

def validate(model,Dataset,Loss,device,normStateOut=None):# NormStateOut is not used
    '''
    Takes model, Dataset, Loss Function, device - Dataset is defined as ProcessingDataset in the Dataset.py
    Iterates over the dataset using the IterateInBatches function
    Returns the average loss
    '''
    # make sure the Dataset State is Val
    Dataset.State = 'Val'
    model.eval()
    batchN = 0 
    with torch.no_grad():
        Pred  = []
        Truth = []
        for DatasetEventIndex,BatchTraces,BatchAux,BatchTruth in IterateInBatches(Dataset,256):
            BatchTraces = BatchTraces          .to(device)
            BatchAux    = BatchAux             .to(device)
            BatchTruth  = BatchTruth           .to(device)
            
            predictions = model(BatchTraces,BatchAux)
            Pred.append(predictions)
            Truth.append(BatchTruth)

        Pred = torch.cat(Pred,dim=0)
        Truth = torch.cat(Truth,dim=0)
        val_loss,val_loss_Phi,val_loss_Theta = Loss(Pred,Truth)
        val_loss = val_loss.item()
        val_loss_Phi = val_loss_Phi.item()
        val_loss_Theta = val_loss_Theta.item()
    # Compute the metrics and unnormalise
    Chi0_Pred_P68 = Percentile68(torch.acos(Truth[:,0]),torch.acos(Pred[:,0]))*180/torch.pi
    Rp_Pred_P68   = Percentile68(5800*      Truth[:,1] ,5800*      Pred[:,1])

    # Just a printout
    metric = f'Validation 68% Chi: {Chi0_Pred_P68:.3f} deg, Rp: {Rp_Pred_P68:.3f} m'

    return val_loss,val_loss_Phi,val_loss_Theta,metric






