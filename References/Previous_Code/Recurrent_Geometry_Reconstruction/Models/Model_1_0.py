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

class Model_1_0(nn.Module):

    def __init__(self,in_channels = 2,in_AuxData = 5,NDenseNodes = 128,NLSTMNodes = 16,LSTMLayers = 1,Dtype = torch.float32):
        
        super(Model_1_0, self).__init__()
        # Info
        self.Name = 'Model_1_0'
        self.Description = '''
        Try to predict Chi0 and Rp using a simple LSTM network with a decoder block
        NOT USING the AuxData Yet
        '''
        self.LSTM1 = nn.LSTM(in_channels, NLSTMNodes, LSTMLayers, batch_first=True,bidirectional=True,dtype=Dtype)
        self.LSTM2 = nn.LSTM(NLSTMNodes*2, NLSTMNodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        
        # Dense Layers
        self.Chi0Dense1 = nn.Linear(NLSTMNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(NLSTMNodes,NDenseNodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        # Activation
        self.DenseActivation  = nn.LeakyReLU()
        
        # # Weight Initialization
        for name, param in self.LSTM1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        for name, param in self.LSTM2.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


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

class Model_1_1(nn.Module):

    def __init__(self,in_channels = 2,in_AuxData = 5,NDenseNodes = 128,NLSTMNodes = 16,LSTMLayers = 1,Dtype = torch.float32):
        
        super(Model_1_1, self).__init__()
        # Info
        self.Name = 'Model_1_1'
        self.Description = '''
        Try to predict Chi0 and Rp using a simple LSTM network with a decoder block
        Using Aux Data now in a dense format
        '''
        self.LSTM1 = nn.LSTM(in_channels, NLSTMNodes, LSTMLayers, batch_first=True,bidirectional=True,dtype=Dtype)
        self.LSTM2 = nn.LSTM(NLSTMNodes*2, NLSTMNodes, 1, batch_first=True,bidirectional=False,dtype=Dtype)
        

        # Aux Analysis
        self.AuxDense1 = nn.Linear(in_AuxData+NLSTMNodes,NDenseNodes,dtype=Dtype)
        self.AuxDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.AuxDense3 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)


        # Decoder Block
        self.Chi0Dense1 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.Chi0Dense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        self.RpDense1 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense2 = nn.Linear(NDenseNodes,NDenseNodes,dtype=Dtype)
        self.RpDense3 = nn.Linear(NDenseNodes,1,dtype=Dtype)

        # Activation
        self.DenseActivation  = nn.LeakyReLU()
        
        # # Weight Initialization
        for name, param in self.LSTM1.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        for name, param in self.LSTM2.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


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
        
        # Predict
        Chi0 = self.DenseActivation(self.Chi0Dense1(out))
        Chi0 = self.DenseActivation(self.Chi0Dense2(Chi0))
        Chi0 = self.Chi0Dense3(Chi0)
        Rp = self.DenseActivation(self.RpDense1(out))
        Rp = self.DenseActivation(self.RpDense2(Rp))
        Rp = self.RpDense3(Rp)
        
        return torch.cat((Chi0,Rp),dim=1)


# Define the Loss Function
    
def Loss(Pred,Truth):

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
        val_loss = 0
        val_loss_Phi = 0
        val_loss_Theta = 0
        for DatasetEventIndex,BatchTraces,BatchAux,BatchTruth in IterateInBatches(Dataset,256):
            BatchTraces = BatchTraces          .to(device)
            BatchAux    = BatchAux             .to(device)
            BatchTruth  = BatchTruth          .to(device)
            
            predictions = model(BatchTraces,BatchAux)
            loss,PhiLoss,ThetaLoss = Loss(predictions,BatchTruth)
            val_loss += loss.item()
            val_loss_Phi += PhiLoss.item()
            val_loss_Theta += ThetaLoss.item()
            batchN += 1
    val_loss = val_loss/batchN
    val_loss_Phi = val_loss_Phi/batchN
    val_loss_Theta = val_loss_Theta/batchN
    return val_loss,val_loss_Phi,val_loss_Theta






