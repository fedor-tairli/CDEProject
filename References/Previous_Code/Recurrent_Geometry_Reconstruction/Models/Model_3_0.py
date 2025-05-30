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
class Model_3_0_Donor(nn.Module):
    # Info
    Name = 'Model_3_0_Donor'
    Description = '''
    Part of the Model_3_X series
    Dummy model, so i dont have to change the code
    '''

    def __init__(self,in_channels,in_AuxData,n_dense_nodes,n_LSTM_nodes,LSTM_layers,dropout_prob,Dtype = torch.float32):
        pass


class Model_3_0_Recipient(nn.Module):
    # Info
    Name = 'Model_3_0_Recipient'
    Description = '''
    Part of the Model_3_X series
    The Recipient Model (This one is trial, has no donor)

    Predict the difference between rec and truth data to improve the prediction
    '''

    def __init__(self,in_channels = 3,in_AuxData = 5,n_dense_nodes = 64,n_LSTM_nodes = 16,LSTM_layers = 3,dropout_prob=0,Dtype = torch.float32):
        LSTM_dropout_prob = 0.5*dropout_prob
        super(Model_3_0_Recipient, self).__init__()
        self.LSTM1 = nn.LSTM(in_channels, n_LSTM_nodes, LSTM_layers, batch_first=True,bidirectional=True,dropout=LSTM_dropout_prob,dtype=Dtype)
        self.LSTM2 = nn.LSTM(n_LSTM_nodes*2, n_LSTM_nodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        # Aux Analysis
        self.AuxDense1 = nn.Linear(in_AuxData+n_LSTM_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense2 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense3 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        # Concatenate the data here
        n_dense_nodes = n_dense_nodes + n_LSTM_nodes + in_AuxData
        self.AuxDense4 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense5 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense6 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        # Concatenate the data here
        n_dense_nodes = n_dense_nodes + n_LSTM_nodes + in_AuxData
        self.AuxDense7 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense8 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)
        self.AuxDense9 = nn.Linear(n_dense_nodes,n_dense_nodes,dtype=Dtype)

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
        LSTM_out = out[:,-1,:]
        out = torch.cat((LSTM_out,AuxData),dim=1)

        # Aux Analysis
        out = self.DenseActivation(self.AuxDense1(out))
        out = self.DenseActivation(self.AuxDense2(out))
        out = self.DenseActivation(self.AuxDense3(out))
        out = self.dropout(out)
        out = torch.cat((out,LSTM_out,AuxData),dim=1)
        out = self.DenseActivation(self.AuxDense4(out))
        out = self.DenseActivation(self.AuxDense5(out))
        out = self.DenseActivation(self.AuxDense6(out))
        out = self.dropout(out)
        out = torch.cat((out,LSTM_out,AuxData),dim=1)
        out = self.DenseActivation(self.AuxDense7(out))
        out = self.DenseActivation(self.AuxDense8(out))
        out = self.DenseActivation(self.AuxDense9(out))
        out = self.dropout(out)

        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = torch.tanh(          self.Chi0Dense3(Chi0))
        
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp =                      self.RpDense3(Rp)
        
        
        
        return torch.cat((Chi0,Rp),dim=1)

    def TransplantWeights(self,DonorModel):
        '''
        Transplant the weights from the Donor Model
        '''
        pass





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
    # # Compute the metrics and unnormalise
    Chi0_Pred_P68 = Percentile68(torch.acos(Truth[:,0]),torch.acos(Pred[:,0]))*180/torch.pi
    Rp_Pred_P68   = Percentile68(5800*      Truth[:,1] ,5800*      Pred[:,1])

    # Just a printout
    metric = f'Validation 68% Chi: {Chi0_Pred_P68:.4f} deg, Rp: {Rp_Pred_P68:.4f} m'
    # metric = ''
    return val_loss,val_loss_Phi,val_loss_Theta,metric






