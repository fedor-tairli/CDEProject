# Importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import numpy as np
import hexagdly
import matplotlib.pyplot as plt
import os


def Loss(Pred,Truth):
    assert Pred.shape[1:] == torch.Size([120,1]), 'Pred shape is {}'.format(Pred.shape)
    assert Truth.shape[1:] == torch.Size([120,1]), 'Truth shape is {}'.format(Truth.shape)

    criterion = nn.MSELoss()
    loss = criterion(Pred,Truth)
    return loss



def validate(model,dataloader,Loss,device = 'cuda',Unnorm=False):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Features, Trace in dataloader:
            Features = Features.to(device)
            Trace = Trace.to(device)
            Y_pred = model(Features)
            
            val_loss += Loss(Y_pred,Trace).item()
    return val_loss/len(dataloader)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, Features, Trace):
        self.Features = Features
        self.Trace = Trace
    def __len__(self):
        return len(self.Features)
    def __getitem__(self, idx):
        return self.Features[idx,:], self.Trace[idx,:]
    


    
class Model_5_TraceGenerator(nn.Module):

    def __init__(self,  hidden_dim=30,features=12):
        super(Model_5_TraceGenerator, self).__init__()
        # Info
        self.Name = 'Model_5_TraceGenerator'
        self.Description = '''
        Instead of trying to guess what the fuck, predict the trace to see what the features do
        Model is very simple
        '''

        # Layers
        self.LSTM        = nn.LSTM(features+1, hidden_dim, batch_first=True)
        self.CollectLSTM = nn.LSTM(hidden_dim,1,batch_first=True)
        self.AvergeTrace = torch.tensor([0.0256, 0.0823, 0.1157, 0.1315, 0.1366, 0.1369, 0.1347, 0.1308, 0.1259,
        0.1205, 0.1146, 0.1086, 0.1028, 0.0970, 0.0913, 0.0858, 0.0805, 0.0756,
        0.0708, 0.0664, 0.0623, 0.0584, 0.0549, 0.0515, 0.0485, 0.0456, 0.0429,
        0.0404, 0.0381, 0.0360, 0.0340, 0.0321, 0.0304, 0.0287, 0.0272, 0.0258,
        0.0245, 0.0233, 0.0222, 0.0211, 0.0201, 0.0191, 0.0182, 0.0173, 0.0165,
        0.0158, 0.0150, 0.0143, 0.0137, 0.0130, 0.0125, 0.0119, 0.0114, 0.0108,
        0.0104, 0.0099, 0.0095, 0.0091, 0.0087, 0.0083, 0.0080, 0.0077, 0.0074,
        0.0071, 0.0068, 0.0065, 0.0062, 0.0060, 0.0058, 0.0055, 0.0053, 0.0050,
        0.0049, 0.0047, 0.0045, 0.0044, 0.0042, 0.0040, 0.0039, 0.0038, 0.0036,
        0.0034, 0.0033, 0.0032, 0.0031, 0.0030, 0.0029, 0.0028, 0.0027, 0.0026,
        0.0025, 0.0024, 0.0023, 0.0022, 0.0021, 0.0021, 0.0020, 0.0019, 0.0018,
        0.0018, 0.0017, 0.0017, 0.0016, 0.0016, 0.0015, 0.0015, 0.0014, 0.0014,
        0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0010,
        0.0010, 0.0010, 0.0009])
        self.AvergeTrace = self.AvergeTrace.unsqueeze(0).unsqueeze(2)
    def forward(self, features):
        # Features would be N,12 in shape therefore need to tile it 120 times to make a shape N,12,120
        
        features = features.unsqueeze(2).repeat(1,1,120).transpose(1,2)
        N = features.shape[0]
        Trace = self.AvergeTrace.repeat(N,1,1).to(features.device)
        out = torch.cat((features,Trace),dim=2)
        
        out, _ = self.LSTM(out)
        out, _ = self.CollectLSTM(out)
        
        return out
